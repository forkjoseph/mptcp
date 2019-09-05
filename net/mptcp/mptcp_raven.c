/*
 *	MPTCP RAVEN: Strategic Redundancy MPTCP Scheduler v0.1
 *
 *	This is from the implementation of MPTCP RAVEN scheduler in
 *	HyunJong Lee, Jason Flinn, and Basavaraj Tonsha,
 *		"RAVEN: Improving Interactive Latency for the Connected Car"
 *	in MobiCom ’18: 24th Annual International Conference on Mobile
 *	Computing and Networking, October 2018.
 *	Available from:
 *		http://leelabs.org/pubs/mobicom18_raven.pdf
 *
 *	Current Maintainer & Author:
 *	HyunJong (Joseph) Lee <hyunjong@umich.edu>
 */
 
#include <net/mptcp.h>
#include <net/mptcp_raven_config.h>
#include <net/mptcp_raven_stale_table.h>

/* #define FLIPFLOP 1 */

static bool has_been_est;
static unsigned int has_been_sent;

static int *lambdas;
static int target_case = 2;
static int target_ci = 90;
/* #if defined(FLIPFLOP) && FLIPFLOP */
static int to_stripe = 0;
static int to_redundancy = 0;
/* #endif */

/* private data */
struct raven_sock_data {
  int pi_idx;
}; 

#define pi_to_flag(sk) mptcp_pi_to_flag(mptcp_path_index(sk))
u8 mptcp_path_index(const struct sock *sk)
{
	return tcp_sk(sk)->mptcp->path_index;
}

int mptcp_dont_reinject_skb(const struct tcp_sock *tp, 
		const struct sk_buff *skb)
{
	return skb && 
		mptcp_pi_to_flag(tp->mptcp->path_index) & TCP_SKB_CB(skb)->path_mask;
}

bool mptcp_is_reinjected(const struct sk_buff *skb)
{
	return TCP_SKB_CB(skb)->mptcp_flags & MPTCP_REINJECT;  
}

bool mptcp_is_skb_redundant(const struct sk_buff *skb)
{
	return TCP_SKB_CB(skb)->mptcp_flags & MPTCP_REDUNDANT;
}

bool mptcp_snd_wnd_test(const struct tcp_sock *tp, 
    const struct sk_buff *skb, unsigned int cur_mss);
unsigned int mptcp_cwnd_test(const struct tcp_sock *tp,
			   const struct sk_buff *skb);
static bool raven_mptcp_is_temp_unavailable(struct sock *sk, const struct sk_buff
    *skb, bool zero_wnd_test);

static struct raven_sock_data *raven_get_data(const struct tcp_sock *tp)
{
	return (struct raven_sock_data *)&tp->mptcp->mptcp_sched[0];
}

static struct sock *get_valid_next_subflow(struct sk_buff *skb, 
    struct mptcp_cb *mpcb) 
{
  struct sock *sk;
  struct tcp_sock *tp;
  u8 pi_bit, done_pi;
  int cnt = 0;

  if (!skb || !mptcp_is_skb_redundant(skb))
    goto fail;

  /* check it's valid target rdn subflow
   * if valid, check whether it's wifi
   * if it's wifi, check other subflows have actually sent the data
   * if not, assign rdn next subflow to next one
   * jump to get_next_subflow
   */
get_next_subflow:
  tp = skb->redundant_next_subflow;
  sk = (struct sock *) tp;

  if (!sk) 
    goto assign_next;

  if (!tp->mptcp || !skb->redundant_target_pis) {
    /* pr_emerg("[%s] rdn_target %u, pi XX, seq %u, \n", __func__, */
    /*     skb->redundant_target_pis, TCP_SKB_CB(skb)->seq); */
    goto fail;
  } 

  /* sk := redundant_next_subflow,
   * if sk valid target?
   *    if sk is NOT wifi? 
   *      return sk
   *    done_pi = skb's rdn path mask
   *    if (done_pi != target - wifi pi)
   *      goto assign_next
   *    return sk
   */
  pi_bit = mptcp_pi_to_flag(mptcp_path_index(sk));
  done_pi = skb->redundant_path_mask;

  /* pr_emerg("target %u, pi %u, done %u\n", */ 
  /*     skb->redundant_target_pis, pi_bit,  done_pi); */

  if ((skb->redundant_target_pis & pi_bit)) {
    /* if not wifi? np! */
    /* if (!nowifi(sk, skb)) { */
    /*   /1* pr_emerg("target %u, pi %u, done %u\n", *1/ */ 
    /*   /1*     skb->redundant_target_pis, pi_bit,  done_pi); *1/ */

    /*   /1* haven't been sent over yet *1/ */
    /*   if (!(done_pi & pi_bit)) */
    /*     goto bye; */
    /* } else */ 
    {
      /* ughhhh wifi, I hate you... */
      done_pi |= pi_bit;

      /* only wifi is left?! */
      if (done_pi == skb->redundant_target_pis)
        goto bye;
    }
  }

  /* pr_emerg("[%s] pi %u is not valid (target %u)\n", __func__, */
  /*     mptcp_path_index(sk), */ 
  /*     skb->redundant_target_pis); */

assign_next:
  if (sk && tp->mptcp ) { 
    skb->redundant_next_subflow = tp->mptcp->next;
  }

  if (!skb->redundant_next_subflow) 
    skb->redundant_next_subflow = mpcb->connection_list;
  cnt++;
  if (cnt++ > 10)
    goto fail;
  goto get_next_subflow;

bye:
#if 0 
  pr_emerg("[%s] rdn_target %u, pi %2u, seq %u\n", __func__,
      skb->redundant_target_pis, mptcp_path_index(sk),
      TCP_SKB_CB(skb)->seq);
#endif 
  return sk;

fail:
  if (cnt > 10) pr_emerg("[%s] loop protection detected...!\n", __func__);
  return NULL;
}

/* clones original skb
 * puts onto the redundant queue
 * returns cloned skb
 */
struct sk_buff *mptcp_raven_enqueue_rdn_skb(struct sk_buff *orig_skb,
    struct sock *sk, unsigned int cnt_subflows, u8 subflow_pis)
{
  struct tcp_sock *tp = tcp_sk(sk);
  struct sock *meta_sk = mptcp_meta_sk(sk);
  struct tcp_sock *meta_tp = tcp_sk(meta_sk);
  struct mptcp_cb *mpcb = meta_tp->mpcb;
  struct sk_buff *skb;
  unsigned int tmp = cnt_subflows - 1;
  const int mptcp_dss_len = MPTCP_SUB_LEN_DSS_ALIGN +
    MPTCP_SUB_LEN_ACK_ALIGN + MPTCP_SUB_LEN_SEQ_ALIGN;

  if (!orig_skb) 
    goto fail;

  /* have noe idea why it's causing crash... but it fixes... */
  if (!skb_shinfo(orig_skb))
    goto fail;

  /* clone & maintain separate skb to prevent rtx mass-up */
  skb = pskb_copy_for_clone(orig_skb, GFP_ATOMIC);
  if (unlikely(!skb)) {
    mptcp_debug("[%s] WARNING! skb is NULL for seq %u\n", __func__,
        TCP_SKB_CB(orig_skb)->seq);
    goto fail;
  }

  /* assuming sending via meta */
  skb->cnt_redundant = tmp;
  orig_skb->cnt_redundant = tmp;

  /* subflow_pis ^= mptcp_pi_to_flag(mptcp_path_index(sk)); */

  skb->redundant_target_pis = subflow_pis;
  orig_skb->redundant_target_pis = subflow_pis;

  /* advance original skb to original version of cloned skb 
   * points next skb to send. Note that by setting reinject > 1,
   * after pushing to ALL subflows' skb queue, it will be freed.
   *
   * _MUST_ set original skb's redundant flag to tell meta_sk 
   * that ignore possible old ACKs 
   */
  tcp_advance_send_head(meta_sk, orig_skb);

  skb->sk = meta_sk;
  TCP_SKB_CB(orig_skb)->mptcp_flags |= MPTCP_REDUNDANT;
  TCP_SKB_CB(skb)->mptcp_flags |= MPTCP_REDUNDANT;
  memset(TCP_SKB_CB(skb)->dss, 0,  mptcp_dss_len);
  skb->redundant_reinjected = false;
  orig_skb->redundant_reinjected = false;

  /* 1) assign next subflow to send over
   * 2) put rdn_skb to red_write_queue
   */
/* assign_next_subflow: */
  skb->redundant_next_subflow = tp->mptcp->next;
  if (!skb->redundant_next_subflow) {
    skb->redundant_next_subflow = mpcb->connection_list;
  }

  skb_queue_tail(&mpcb->raven_write_queue, skb);
  mpcb->cnt_redundant_skb++;

  if ((TCP_SKB_CB(orig_skb)->tcp_flags & TCPHDR_PSH))
    TCP_SKB_CB(skb)->tcp_flags |= TCPHDR_PSH;

#if 0
  u32 seq, eseq;
  seq = TCP_SKB_CB(skb)->seq;
  eseq = TCP_SKB_CB(skb)->end_seq;
  if (unlikely(sysctl_mptcp_raven_debug))
    mptcp_debug("[%s] redundant queue %u, len %u [%u - %u], curr %u, spi %u\n",
        __func__, skb_queue_len(&mpcb->raven_write_queue), skb->len,
        seq, eseq, atomic_read(&skb->cnt_redudnant),   skb->redundant_target_pis
      );
#endif 

  return skb;

fail:
  return NULL;
}

/*
 * @returns 
 *  -1 : don't even consider it (error)
 *   0 : don't use it (unknown)
 *   1 : not enough samples
 *   2 : use it with PI overlap check
 *   3 : use it without PI overlap check
 */
int mptcp_raven_build_pi(struct sock *sk, u64 *ret_wmean, u64 *ret_wsep,
    u64 *ret_ess, s64 *ret_min_delta, u64 *ret_wsum, ktime_t now_ts)
{
  const struct tcp_sock *tp = tcp_sk(sk);
  struct raven_pim_stat *stat = tp->mptcp->pim_stat;

  int ret = 0;
  int cnt_samples = 0;

  u64 wsum = 0;
  u64 prediction = 0;

  s64 min_delta = 0x0FFFFFFF;

  /* TCP is close state, why bother to us PI? */
  if ((1 << (tp->meta_sk->sk_state))& (TCPF_CLOSE_WAIT | TCPF_LAST_ACK))
    goto failed;

  /* the prediction is:
   *    weighted_sum
   *  -------------------
   *      weight
   */
  stat->last_pred_ts = now_ts;
  if (!stat->last_pred_exist) {
    stat->last_pred_exist = true;
  }

  if (!stat->norm_weight) {
    ret = 3;
    goto done;
  }

  prediction = (stat->norm_wsum * 1000000) / stat->norm_weight;
  prediction /= 1000000;
  ret = 2;

done:
  *ret_wmean = prediction;
  if (stat && stat->last_msmt_exist) { 
    min_delta = ktime_to_ms(ktime_sub(now_ts, stat->last_msmt_ts));
  }

  if (ret == 2) {
    *ret_wsep = 0;
    /* *ret_ess = ess; */
    *ret_ess = cnt_samples;
    *ret_min_delta = min_delta;
    *ret_wsum = wsum;
    return ret;
  } else if (ret == 3) { 
    *ret_wsep = 0;
    *ret_ess = 0;
    *ret_min_delta = 0;
    return 3;
  }

failed:
  *ret_wmean = 0;
  *ret_wsep = 0;
  *ret_ess = 0;
  return -1;
}

int mptcp_raven_get_pi(struct sock *sk, u64 wmean, s64 min_delta, 
    u64 *pi_min, u64 *pi_max, ktime_t now_ts) 
{
  struct tcp_sock *tp = tcp_sk(sk);
  struct raven_sock_data *rsd = raven_get_data(tp);
  int pi_idx = rsd->pi_idx;
  
  u64 l_bound, u_bound;
  int aging_idx = min_delta / 1000;
  u64 aging = 1000000;
  int tmp_aging;
  u64 decay = 1000000;
  s64 tmp;

  if (pi_idx < 0) {
    l_bound = 0;
    u_bound = ULONG_MAX / 2;
    pr_emerg("[%s] pi %u's pi_idx not assigned!\n", __func__,
        mptcp_path_index(sk));
    goto out;
  }

  if ((aging_idx < 0 || aging_idx > 1000)) { 
    tmp_aging = aging_idx;
    if (pi_idx < 2) {
      aging_idx = 1000;
    } else { 
      /* max idx for aging factors */
      if (target_case == 1) {
        aging_idx = 35; 
      } else if (target_case == 4) {
        aging_idx = 12;
      }
    }

    /* pr_emerg("[%s] pi %u, pi_idx %u, md %lld, aging idx from %d to %d, " */
    /*     "nts %llu, mts %llu" */
    /*     "\n", __func__, mptcp_path_index(sk), pi_idx, min_delta, */
    /*     tmp_aging, aging_idx, ktime_to_ms(now_ts), */
    /*     ktime_to_ms(stat->last_msmt_ts) */
    /*   ); */
  }

  if (target_case == 1) { 
		if (unlikely(sysctl_mptcp_raven_aging)) {
			if (pi_idx == 1) {
				aging = D1_1_stale[aging_idx];
			} else if (pi_idx == 2) {
				aging = D1_2_stale[aging_idx];
			} else if (pi_idx == 3) { 
				aging = D1_3_stale[aging_idx];
			}
		}

    u_bound = (D1_pi_upper[pi_idx] * aging) / 1000000;
    u_bound = (decay + u_bound) / 1000000;
    u_bound = wmean * u_bound;

    l_bound = (D1_pi_lower[pi_idx] * aging) / 1000000;
    tmp = (decay - l_bound);
    tmp = (wmean * tmp);
    l_bound = tmp / 1000000;
  } else if (target_case == 2) {
    if (mptcp_path_index(sk) == 1) { 
      aging = D2_1_stale[aging_idx];
    } else if (mptcp_path_index(sk) == 2) { 
      aging = D2_2_stale[aging_idx];
    }

    u_bound = (D2_pi_upper[pi_idx] * aging) / 1000000;
    u_bound = (decay + u_bound) / 1000000;
    u_bound = wmean * u_bound;

    l_bound = (D2_pi_lower[pi_idx] * aging) / 1000000;
    tmp = (decay - l_bound);
    tmp = (wmean * tmp);
    l_bound = tmp / 1000000;
  } else if (target_case == 3) {
    if (mptcp_path_index(sk) == 1) { 
      aging = D3_1_stale[aging_idx];
    } else if (mptcp_path_index(sk) == 2) { 
      aging = D3_2_stale[aging_idx];
    }

    u_bound = (D3_pi_upper[pi_idx] * aging) / 1000000;
    u_bound = (decay + u_bound) / 1000000;
    u_bound = wmean * u_bound;

    l_bound = (D3_pi_lower[pi_idx] * aging) / 1000000;
    tmp = (decay - l_bound);
    tmp = (wmean * tmp);
    l_bound = tmp / 1000000;
  } else if (target_case == 4) {
    if (mptcp_path_index(sk) == 1) { 
      aging = D4_1_stale[aging_idx];
    } else if (mptcp_path_index(sk) == 2) { 
      aging = D4_2_stale[aging_idx];
    } else if (mptcp_path_index(sk) == 3) { 
      aging = D4_3_stale[aging_idx];
    }

    u_bound = (D4_pi_upper[pi_idx] * aging) / 1000000;
    u_bound = (decay + u_bound) / 1000000;
    u_bound = wmean * u_bound;

    l_bound = (D4_pi_lower[pi_idx] * aging) / 1000000;
    tmp = (decay - l_bound);
    tmp = (wmean * tmp);
    l_bound = tmp / 1000000;
  } else {
    l_bound = 0;
    u_bound = ULONG_MAX / 2;
	}

  *pi_min = l_bound;
  *pi_max = u_bound;
  /* pr_emerg("[%s] pi %u, l %llu, u %llu\n", __func__, */
  /*     pi_idx + 1, l_bound, u_bound); */

out:
  return 0;
}
    

static bool raven_redundant_pi(struct sk_buff *skb, struct tcp_sock *tp)
{
  if (unlikely(!skb || !tp)) return false;

  /* mptcp_write_xmit masks path_mask to path_index (pi) via
   * mptcp_skb_entail 
   */
  if (TCP_SKB_CB(skb)->path_mask & 
      mptcp_pi_to_flag(tp->mptcp->path_index)) 
    return true;
  return false;
}

/* ================ generic stuff ======================= */
struct sock *get_generic_subflow(struct mptcp_cb *mpcb, 
    struct sk_buff *skb, bool (*selector)(struct sock*, struct sk_buff *skb), 
    bool zero_wnd_test, bool *force)
{
 struct sock *bestsk = NULL;
 u32 min_srtt = 0xffffffff;
 bool found_unused = false;
 bool found_unused_una = false;
 struct sock *sk;

 mptcp_for_each_sk(mpcb, sk) {
   struct tcp_sock *tp = tcp_sk(sk);
   bool unused = false;

   /* first, we choose only the wanted sks */
   if (!(*selector)(sk, skb))
     continue;

   if (!mptcp_dont_reinject_skb(tp, skb))
     unused = true;
   else if (found_unused)
     /* if a unused sk was found previously, we continue -
      * no need to check used sks anymore.
      */
     continue;

   if (mptcp_is_def_unavailable(sk))
     continue;

/* MICRO: RMSE for JS */
    if (mptcp_path_index(sk) == 1) {
     mpcb->pi1_predicted = (tp->srtt_us >> 3);
    } else if (mptcp_path_index(sk) == 2) {
     mpcb->pi2_predicted = (tp->srtt_us >> 3);
    } else if (mptcp_path_index(sk) == 3) {
     mpcb->pi3_predicted = (tp->srtt_us >> 3);
    }
  

   if (raven_mptcp_is_temp_unavailable(sk, skb, zero_wnd_test)) {
     if (unused)
       found_unused_una = true;
     continue;
   }

   if (unused) {
     if (!found_unused) {
       /* it's the first time we encounter an unused
        * sk - thus we reset the bestsk (which might
        * have been set to a used sk).
        */
       min_srtt = 0xffffffff;
       bestsk = NULL;
     }
     found_unused = true;
   }

   if (tp->srtt_us < min_srtt) {
     min_srtt = tp->srtt_us;
     bestsk = sk;
   }
 }

 if (bestsk) {
   /* the force variable is used to mark the returned sk as
    * previously used or not-used.
    */
   if (found_unused)
     *force = true;
   else
     *force = false;
 } else {
   /* the force variable is used to mark if there are temporally
    * unavailable not-used sks.
    */
   if (found_unused_una)
     *force = true;
   else
     *force = false;
 }

 return bestsk;
}

/* this check whether this subflow is available or not
 * and does not have global view
 */
static bool raven_mptcp_is_temp_unavailable(struct sock *sk,
				      const struct sk_buff *skb,
				      bool zero_wnd_test)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	unsigned int mss_now, space, in_flight;

  if (!tp || !tp->mptcp) 
    return true;

	if (!tp->mptcp->fully_established) {
		/* make sure that we send in-order data */ 
		if (skb && tp->mptcp->second_packet &&
		    tp->mptcp->last_end_data_seq != TCP_SKB_CB(skb)->seq) {
			return true;
    }
	}

	/* if tsq is already throttling us, do not send on this subflow. when
	 * tsq gets cleared the subflow becomes eligible again.
	 */
  // joseph: tsq ''forces'' subflow2 to use
#if 0 
	if (test_bit(tsq_throttled, &tp->tsq_flags)) {
      u32 limit = max(2 * skb->truesize, sk->sk_pacing_rate >> 10);
      pr_emerg("tsq_throttled for %d [%u] [%u]\n", tp->mptcp->path_index, 
          limit, atomic_read(&sk->sk_wmem_alloc));
		return true;
  }
#endif

	in_flight = tcp_packets_in_flight(tp);
	/* not even a single spot in the cwnd */
	if (in_flight >= tp->snd_cwnd) {
		return true;
  }

	/* now, check if what is queued in the subflow's send-queue
	 * already fills the cwnd.
	 */
	space = (tp->snd_cwnd - in_flight) * tp->mss_cache;

	if (tp->write_seq - tp->snd_nxt > space) { 
		return true;
  }

	if (zero_wnd_test && !before(tp->write_seq, tcp_wnd_end(tp))) {
		return true;
  }

	mss_now = tcp_current_mss(sk);

	/* don't send on this subflow if we bypass the allowed send-window at
	 * the per-subflow level. similar to tcp_snd_wnd_test, but manually
	 * calculated end_seq (because here at this point end_seq is still at
	 * the meta-level).
	 */
	if (skb && !zero_wnd_test &&
	    after(tp->write_seq + min(skb->len, mss_now), tcp_wnd_end(tp))) 
  {
		return true;
  }

	return false;
}

static bool dummy_selector(struct sock* sk, struct sk_buff *skb) { return true; }

/* redundancy subflow selector */
struct sock *get_redundant_subflow(struct mptcp_cb *mpcb, 
      struct sk_buff *skb, bool (*selector)(struct sock*, struct sk_buff*), 
      bool zwnd_test, bool *force)
{
	struct sock *sk, *sk2, *bestsk = NULL, *meta_sk = mpcb->meta_sk;
  struct sk_buff *rdn_skb = NULL;
  struct tcp_sock *meta_tp = tcp_sk(meta_sk);

  /* statistics stuff */ 
	u32 min_srtt = 0xffffffff;
  u64 min_high = 0xffffffff, min_low = 0xffffffff;
  int min_idx = 0;
  bool has_minpi_found = false;

  int cnt_subflows = 0;
  u8 subflow_pis = 0;
  u8 subflow_pis_idx = 0;
  u8 target_subflow_pis = 0;
  bool use_redundancy = false;
  bool blind_redundancy = false;
  bool pr_redundancy = false;

  int preventloop = 0;
  ktime_t now_ts;

  if (unlikely(!skb)) {
    return (struct sock *)mpcb->connection_list;
  }

	now_ts = ktime_get();

	// TODO: move to swtiching sysctl 
  if (sysctl_mptcp_raven_measure) 
  {
#if defined(FLIPFLOP) && FLIPFLOP
  int avail_subflows = mpcb->cnt_subflows;
  /* pr_emerg("current mode %s, cnt_sub %u\n", */
  /*     (mpcb->stripe_mode ? "stripe" : "redundancy"), */ 
  /*     avail_subflows); */

  if (avail_subflows > 1 && !mpcb->stripe_mode) {
    mptcp_for_each_sk(mpcb, sk) {
      struct tcp_sock *tp = tcp_sk(sk);
      unsigned int in_flight;
      /* unsigned int mss_now, space; */

      if (!mptcp_sk_can_send(sk)) {
        avail_subflows--;
        continue;
      }

      if (!tp->mptcp->fully_established) {
        /* make sure that we send in-order data */ 
        if (skb && tp->mptcp->second_packet &&
            tp->mptcp->last_end_data_seq != TCP_SKB_CB(skb)->seq) {
          avail_subflows--;
          continue;
        }
      }

      in_flight = tcp_packets_in_flight(tp);
      /* not even a single spot in the cwnd */
      if (in_flight >= tp->snd_cwnd) {
        avail_subflows--;
        /* pr_emerg("pi %d [fli %u, cwnd %u], ", mptcp_path_index(sk), */
        /*     in_flight, tp->snd_cwnd); */
        continue;
      }

      /* now, check if what is queued in the subflow's send-queue
       * already fills the cwnd.
       */
      /* space = (tp->snd_cwnd - in_flight) * tp->mss_cache; */
      /* if (tp->write_seq - tp->snd_nxt > space) { */ 
      /*   avail_subflows--; */
      /*   continue; */
      /* } */

      /* if (!before(tp->write_seq, tcp_wnd_end(tp))) { */
      /*   avail_subflows--; */
      /*   continue; */
      /* } */

      /* mss_now = tcp_current_mss(sk); */
      /* if(after(tp->write_seq + min(skb->len, mss_now), tcp_wnd_end(tp))) { */
      /*   avail_subflows--; */
      /*   continue; */
      /* } */
    }
    /* printk("\n"); */

    /* N - 1 subflows are full! */ 
    if (avail_subflows <= 1) { 
      mpcb->stripe_mode = true;
      to_stripe += 1;
      pr_emerg("[%s] switching to stripe: sb %u, qlen %u, statidx %u\n", __func__,
          avail_subflows, skb_queue_len(&((meta_sk)->sk_write_queue)), stat_idx);
    } else {
      /* pr_emerg("[%s] current avail subflows %u\n", __func__, */
      /*     avail_subflows); */
    }
    /* else { */ 
    /*   mpcb->stripe_mode = false; */
    /* } */
  }

  if (mpcb->stripe_mode) {
    cnt_subflows = avail_subflows;
    if (mptcp_is_skb_redundant(skb))
      goto rdn_detected;
    use_redundancy = false;

    goto get_generic;
  }
#endif
  }

  /* if reinjected, don't use redundancy */
  if (mptcp_is_reinjected(skb)) {
    /* reinject on exact sample pi */
    mptcp_for_each_sk(mpcb, sk) {
      struct tcp_sock *tp = tcp_sk(sk);
      int tmp_cnt_redundant = 0;
      if (!mptcp_sk_can_send(sk)) 
        continue;

#if 0
      if (mpcb->raven_policy->redundancy && sysctl_mptcp_raven_measure) 
      {
    
        if (!nowifi(sk, skb)) {
            skb->cnt_redundant = tmp_cnt_redundant;
            return sk;
        }
      } else 
#endif
      { 
        if (TCP_SKB_CB(skb)->path_mask &
            mptcp_pi_to_flag(tp->mptcp->path_index)) 
        skb->cnt_redundant = tmp_cnt_redundant;
        return sk;
      }
    }
  }

  /* skb is from redundancy write queue? 
   * return next subflow to send redundant data.
   */ 
  if (mptcp_is_skb_redundant(skb)) 
  {
    struct tcp_sock *tp;
rdn_detected:
    bestsk = (struct sock *) skb->redundant_next_subflow;
    tp = tcp_sk(bestsk);

    if (likely(bestsk && tp->mptcp))
    { 
      if (!tp->mptcp->fully_established) 
      {
        /* a change was already given... */
        if (tp->mptcp->second_packet && 
            tp->mptcp->last_end_data_seq != TCP_SKB_CB(skb)->seq) 
        {
          if (tp->mptcp->next) {
            pr_emerg("subflow %u is not fully est. but has next %u, cnt_est %u\n", 
                mptcp_path_index(bestsk),
                mptcp_path_index((struct sock *) tp->mptcp->next),
                mpcb->cnt_established);
            skb->redundant_next_subflow = tp->mptcp->next;
          } else {
            pr_emerg("subflow %u is not fully est. no next \n", mptcp_path_index(bestsk));
          }
          return NULL;
        } 
      } /* not fully est */

      /* leave a mark to say skb has been sent over which subflows */
      skb->redundant_path_mask |= mptcp_pi_to_flag(mptcp_path_index(bestsk));
      skb->redundant_next_subflow = tp->mptcp->next;

      if (mptcp_is_reinjected(skb))
        TCP_SKB_CB(skb)->mptcp_flags ^= MPTCP_REDUNDANT;
      return bestsk;
    }

    // TODO: add return statement here?
    if (mptcp_is_reinjected(skb)) {
      struct sk_buff *tmpskb = tcp_send_head(meta_sk);
      u32 tmpskb_seq = 0;
      if (tmpskb) 
        tmpskb_seq = TCP_SKB_CB(tmpskb)->seq;

      pr_crit("[%s] fix? rdn %s, "
        "seq %u, len %u, pm %u, rpm %u, "
        "meta [snd_una %u, snt_nxt %u, redundant_snd_nxt %u], "
        "wqueue %u"
        "\n"
        , __func__, (mptcp_is_skb_redundant(skb) ? "T" : "F"),
        TCP_SKB_CB(skb)->seq, skb->len, 
        TCP_SKB_CB(skb)->path_mask, skb->redundant_path_mask, 
        meta_tp->snd_una, meta_tp->snd_nxt, meta_tp->redundant_snd_nxt,
        tmpskb_seq
        );
      return NULL;
    }
  }

  /* if (unlikely(!sysctl_mptcp_raven_measure)) */
  /*   return get_generic_subflow(mpcb, skb, &constr_selector, zwnd_test, force); */

  /* at this point, skb must only be from meta's write queue */
  if (!sysctl_mptcp_raven_collect_samples) { 
    /* blind redundancy */
    mptcp_for_each_sk(mpcb, sk) {
      struct tcp_sock *tp = tcp_sk(sk);
      if (!mptcp_sk_can_send(sk)) 
        continue;
      target_subflow_pis |= mptcp_pi_to_flag(tp->mptcp->path_index);
    }

    if (target_subflow_pis) {
      use_redundancy = true;
      blind_redundancy = true;
    }
  } else {
    use_redundancy = true;
    /* Strategic Redundancy algorithm implementation 
     * 
     *  1. compute prediction interval of each subflow
     *  2. given prediction intervals, check whether some of them are
     *    overlapping. if so, make sure to log which ones are overlapping.
     *
     * Note that this step does not check cwnd availability. Later on,
     * check given target subflow, check for availability of those and
     * mask them.
     */
    mptcp_for_each_sk(mpcb, sk) {
      struct tcp_sock *tp = tcp_sk(sk);
      int use_intervals = 0;
      u32 updated = 1; 
      u64 ess;
      u64 wmean, wsep;
      u64 pi_min, pi_max;
      s64 min_delta;
      u64 wsum;
      u32 js_mean; /* Jacobson's mean for comparison */

      /* -1 : never use redundancy
       *  0 : never use redundancy (unknown reason)
       *  1 : use it w/o intervals
       *  2 : use it with intervals
       *  3 : interval is not computed
       */
      use_intervals = mptcp_raven_build_pi(sk, &wmean, &wsep, &ess,
          &min_delta, &wsum, now_ts);

      if (use_intervals == 2) 
				mptcp_raven_get_pi(sk, wmean, min_delta, &pi_min, &pi_max,
						now_ts); 

      js_mean = (tp->srtt_us >> 3);
      /* pr_emerg("pi %u: jacobson %4u, raven: %4llu [%4llu - %4llu] use %d, mDelta %lld\n", */
      /*     mptcp_path_index(sk), js_mean, wmean, pi_min, pi_max, */
      /*     use_intervals, min_delta); */

      if (pi_min < 1000 && pi_max < 1000) {
        /* pr_emerg("[%s] pi %u: raven: %4llu [%4llu - %4llu] use %d, mDelta %lld, wsum %llu\n", */
        /*     __func__, mptcp_path_index(sk), wmean, pi_min, pi_max, */
        /*     use_intervals, min_delta, wsum); */
      }

      if (pi_min > pi_max && use_intervals == 2) { 
        /* pr_emerg("[%s] pi %u: raven: %4llu [%llu - %4llu] use %d, " */
        /*     "mDelta %lld, wsum %llu\n", __func__, mptcp_path_index(sk), */
        /*     wmean, pi_min, pi_max, use_intervals, min_delta, wsum); */
        pi_min = 1;
      }

      if (use_intervals == 2) 
      { 
        rcu_read_lock();
        spin_lock(&tp->mptcp->stat_lock);

        tp->mptcp->stat.wmean_us = wmean;
        tp->mptcp->stat.pi_low = pi_min;
        tp->mptcp->stat.pi_high = pi_max;
        /* tp->mptcp->stat.pi_tail = wme; */
        tp->mptcp->stat.ess = ess;
        tp->mptcp->stat_valid = true;
        tp->mptcp->stat.observed = updated;

        spin_unlock(&tp->mptcp->stat_lock);
        rcu_read_unlock();
      } else if (use_intervals == 3) {
        rcu_read_lock();
        spin_lock(&tp->mptcp->stat_lock);

        tp->mptcp->stat.wmean_us = ULONG_MAX / 2;
        tp->mptcp->stat.pi_low = 0;
        tp->mptcp->stat.pi_high = ULONG_MAX;
        tp->mptcp->stat.pi_tail = ULONG_MAX  / 2;
        tp->mptcp->stat.ess = 0;
        tp->mptcp->stat_valid = true;
        tp->mptcp->stat.observed = updated;

        spin_unlock(&tp->mptcp->stat_lock);
        rcu_read_unlock();
      }
    }

    has_minpi_found = false;
    mptcp_for_each_sk(mpcb, sk) {
      struct tcp_sock *tp = tcp_sk(sk);
      u64 low = tp->mptcp->stat.pi_low;
      u64 high = tp->mptcp->stat.pi_high;

      /* first, dont' consider [0, inf] case */
      if (low > 0 && low < min_low) {
        min_low = low;
        min_high = high;
        has_minpi_found = true;
        min_idx = mptcp_path_index(sk);
      }
    }

    if (!has_minpi_found) {
      use_redundancy = true;
      blind_redundancy = true;
    } else { 

      /* detect interval overlapping */
      mptcp_for_each_sk(mpcb, sk) {
        /* struct tcp_sock *tp = tcp_sk(sk); */

        if (min_idx == mptcp_path_index(sk)) {
          target_subflow_pis |= pi_to_flag(sk);

          mptcp_for_each_sk(mpcb, sk2) {
            struct tcp_sock *tp2 = tcp_sk(sk2);
            u64 low2 = tp2->mptcp->stat.pi_low;
            /* u64 high2 = tp2->mptcp->stat.pi_high; */

            if (mptcp_path_index(sk) == mptcp_path_index(sk2))
              continue;

            if ( (low2 <= min_high)) {
              /* if (mptcp_cwnd_test(tp2, skb)) */
              target_subflow_pis |= pi_to_flag(sk2);
            } else {
              /* pr_emerg(" NOPE! pi %u [%llu - %llu], min_high %llu\n", */ 
              /*     mptcp_path_index(sk2), low2, high2, min_high); */
            }
          }
        }
      }

      if (target_subflow_pis) {
        /* pr_emerg("[%s] %u, minidx %u, ", __func__, target_subflow_pis, min_idx); */

        mptcp_for_each_sk(mpcb, sk) { 
          /* struct tcp_sock *tp = tcp_sk(sk); */
          /* u64 low = tp->mptcp->stat.pi_low / 1000; */
          /* u64 high = tp->mptcp->stat.pi_high / 1000; */
          /* u64 mean = tp->mptcp->stat.wmean_us / 1000; */
          /* u32 js_mean = (tp->srtt_us >> 3) / 1000; */

          /* pr_emerg("pi %u js %4lu, rv %4llu [%4llu - %4llu]\n", */
          /*     mptcp_path_index(sk), js_mean, mean, low, high); */
        }
        /* printk("\n"); */

        blind_redundancy = false;
        use_redundancy = true;
        pr_redundancy = true;
      }
    }
  }

  /* fixme: 
   * 1. check which redundancy (blind vs. pr) is selected
   * 2. for each subflow that is **available** (i.e., cwnd/queue in a
   * subflow has a room to send) and is indeed target subflow, use
   * subflow_pis to mask them.
   **/ 
filter_subflows:
  cnt_subflows = 0;
  preventloop++;

  mptcp_for_each_sk(mpcb, sk) {
    struct tcp_sock *tp = tcp_sk(sk);
    struct raven_sock_data *rsd = raven_get_data(tp);
    bool unused = false;
    u8 pi = pi_to_flag(sk);
    int pi_idx = rsd->pi_idx;

    if (pi_idx < 0)
      pi_idx = 0;
    else
      pi_idx = (1 << pi_idx);

    if (!mptcp_sk_can_send(sk)) {
      pr_crit("[%s] subflow %u can't send\n", __func__,
          mptcp_path_index(sk));
      continue;
    }

    if (use_redundancy && !blind_redundancy) {
      if (!(pi & target_subflow_pis)) {
        continue;
      }
    }

    if (!mptcp_dont_reinject_skb(tp, skb))
      unused = true;

    if (raven_mptcp_is_temp_unavailable(sk, skb, zwnd_test)) {
      /* pr_crit("[%s] subflow %u is temp unavailable\n", __func__, */
      /*     mptcp_path_index(sk)); */
      /* if (!blind_redundancy) { */ 
        /*   continue; */
      /* } */
      continue;
    }

    if (tp->srtt_us < min_srtt) {
      if (blind_redundancy ) {
        bestsk = sk;
      }
    }

    cnt_subflows++;
    subflow_pis |= pi;
    subflow_pis_idx |= pi_idx;
  }

  {
    if (!cnt_subflows && !blind_redundancy) {
      blind_redundancy = true;
      pr_crit("hahahahahaha stupid! %u, skb %u\n", preventloop, TCP_SKB_CB(skb)->seq);
      if (preventloop < 10)
        goto filter_subflows;
    } else if (!cnt_subflows) {
      /* MICRO: flipflop switching b/w striping and redundancy */
      /* pr_crit("potential bug... subflow skb %u, snd_una %u, snd_nxt %u, " */
      /*     "rdn_sndnxt %u, loop %u, pm %u, rpm %u, rdn %s", */
      /*     TCP_SKB_CB(skb)->seq, meta_tp->snd_una, meta_tp->snd_nxt, */
      /*     meta_tp->redundant_snd_nxt, preventloop, */ 
      /*     TCP_SKB_CB(skb)->path_mask, skb->redundant_path_mask, */
      /*     (mptcp_is_skb_redundant(skb) ? "T" : "F") */
      /*     ); */

      if (bestsk) {
        pr_emerg(", best %u\n", mptcp_path_index(bestsk));
      } else {
        bestsk = (struct sock*)mpcb->connection_list;
/* #if !defined(FLIPFLOP) && !FLIPFLOP */
/*         while (mptcp_path_index(bestsk) != 1) { */
/*           bestsk = (struct sock*)tcp_sk(bestsk)->mptcp->next; */ 
/*         } */
/*         cnt_subflows++; */
/*         tcp_sk(bestsk)->snd_cwnd++; */
/*         /1* pr_emerg(", NO BEST assigned %u\n", mptcp_path_index(bestsk)); *1/ */
/*         return bestsk; */
/* #endif */
        return bestsk;
      }

#if defined(FLIPFLOP) && FLIPFLOP
      mpcb->stripe_mode = true;
      pr_emerg("[%s] mode %s, skb %u\n", __func__, (mpcb->stripe_mode ? "stripe" : "redundant"),
          TCP_SKB_CB(skb)->seq);
      return get_generic_subflow(mpcb, skb, &nowifi, zwnd_test, force);
#endif
    }
  }

#if defined(FLIPFLOP) && FLIPFLOP
  /* if (port == 48081)  // flipflop on data-flow only */
  { 
    if (mpcb->stripe_mode) { 
      return get_generic_subflow(mpcb, skb, &nowifi, zwnd_test, force);
      /* return bestsk; */
      /* return get_generic_subflow(mpcb, skb, &constr_selector, zwnd_test, force); */
    }
  }
#endif

  if (cnt_subflows == 1) {
    /* if (blind_redundancy) */ 
      /* pr_crit("[%s] one subflow is available pi bit %u, skb %u, mode %s\n", __func__, */
      /*     subflow_pis, TCP_SKB_CB(skb)->seq, (mpcb->stripe_mode ? "stripe" : "redundancy")); */

    /* use_redundancy = false; */

    preventloop = 0;
    /* while (mptcp_path_index(bestsk) != 1) { */ 
    /*   bestsk = (struct sock*) tcp_sk(bestsk)->mptcp->next; */
    /*   if (!bestsk) */ 
    /*     bestsk = (struct sock *)mpcb->connection_list; */
    /*   preventloop++; */
    /*   /1* if (preventloop > 5) *1/ */
    /*   /1*   break; *1/ */
    /* } */

    if (likely(bestsk)) {
      int set_cntrdn = 0;
      /* atomic_set(&skb->cnt_redundant, set_cntrdn); */
      skb->cnt_redundant = set_cntrdn;
      /* pr_crit("[%s] one subflow is available pi bit %u, seq %u, len %4u, " */
      /*     "snd_una %u, snd_nxt %u, redundant_snd_nxt %u\n", __func__, */
      /*     subflow_pis, TCP_SKB_CB(skb)->seq, skb->len, */
      /*     tcp_sk(meta_sk)->snd_una, tcp_sk(meta_sk)->snd_nxt, */
      /*     tcp_sk(meta_sk)->redundant_snd_nxt); */

      tcp_sk(mptcp_meta_sk(bestsk))->redundant_snd_nxt = TCP_SKB_CB(skb)->end_seq;
    } else {
      /* pr_crit("[%s] one subflow is available pi bit %u, skb %u\n", __func__, */
      /*     subflow_pis, TCP_SKB_CB(skb)->seq); */
    }

    return bestsk;
  }

  /* for fin, don't use redundancy, we don't care */
  if (unlikely(TCP_SKB_CB(skb)->mptcp_flags & MPTCPHDR_FIN)) 
    return bestsk;

  if (unlikely(mptcp_is_data_fin(skb)) || 
      unlikely(meta_sk->sk_shutdown == SHUTDOWN_MASK) ||
      cnt_subflows < 1)
  {
    /* use_redundancy = false; */
    return NULL;
  }

  /* decided to use redundancy (can be either blind or principled 
   * either way, what subflows to send redundantly?
   */
  if (use_redundancy && has_been_sent > 2) {
    if (mpcb->cnt_established > 1 && skb->len > 1) {
      /* pr_emerg("bestsk %u, target %u\n", pi_to_flag(bestsk), subflow_pis); */

      /* put into redundant queue to retrieve later */
      rdn_skb = mptcp_raven_enqueue_rdn_skb(skb, bestsk, cnt_subflows, subflow_pis);
      if (unlikely(!rdn_skb))
        return NULL;

      /* you know, just another logging */
      skb->redundant_path_mask |= mptcp_pi_to_flag(tcp_sk(bestsk)->mptcp->path_index);
      rdn_skb->redundant_path_mask |= mptcp_pi_to_flag(tcp_sk(bestsk)->mptcp->path_index);

      /* to handle networks w/ different speed */
      tcp_sk(mptcp_meta_sk(bestsk))->redundant_snd_nxt = TCP_SKB_CB(skb)->end_seq;

      return bestsk;
    }
  }

  /* at the end of day, falls to generic algorithm when redundancy is
   * not employed (e.g., one subflow available) */
	return bestsk;
}

/* 
 * this is the scheduler. this function decides on which flow to send a given
 * mss. if all subflows are found to be busy, NULL is returned the flow is
 * selected based on the shortest rtt.  if all paths have full cong windows, we
 * simply return NULL.
 * additionally, this function is aware of the backup-subflows.
 */
struct sock *raven_get_subflow(struct sock *meta_sk, 
    struct sk_buff *skb, bool zero_wnd_test)
{
	struct sock *sk = NULL;
  struct sock *ret = NULL;
  struct tcp_sock *meta_tp = tcp_sk(meta_sk);
	struct mptcp_cb *mpcb = meta_tp->mpcb;
	bool force;

  if (unlikely(zero_wnd_test))
    pr_crit("called from %pS : cnt_subflows %u, "
        " cnt_est %u\n", __builtin_return_address(0),
        mpcb->cnt_subflows, mpcb->cnt_established);

	/* if there is nly one subflow, bypass the scheduling function */
	if ((meta_sk->sk_shutdown ^ RCV_SHUTDOWN)
      && mpcb->cnt_established == 1 && !has_been_est) 
  {
		sk = (struct sock *)mpcb->connection_list;
    /* pr_crit("only 1 avialable...!\n"); */
		if (!mptcp_is_available(sk, skb, zero_wnd_test))
			sk = NULL;
    ret = sk;
    goto out;
	}

	/* answer data_fin on same subflow!!! */
  if (meta_sk->sk_shutdown & RCV_SHUTDOWN && skb &&
      mptcp_is_data_fin(skb)) 
  {
		mptcp_for_each_sk(mpcb, sk) {
      if (tcp_sk(sk)->mptcp->path_index == mpcb->dfin_path_index &&
          mptcp_is_available(sk, skb, zero_wnd_test)) 
      {
        ret = sk;
        goto out;
      }
    }
	}

  if (skb) {
    if (!has_been_sent)
      mpcb->rdn_init_seq = 0;
    has_been_sent += skb->len;
    if (!has_been_est) 
    {
      int fully_cnt = 0;
      mptcp_for_each_sk(mpcb, sk) {
        struct tcp_sock *tp = tcp_sk(sk);
        if (tp->mptcp->fully_established)
          fully_cnt++;
      }
      if (fully_cnt > 1)
        has_been_est = true;
      else
        has_been_est = false;
    }
  }

  if (!has_been_est) { 
    sk = get_generic_subflow(mpcb, skb, &dummy_selector, zero_wnd_test,  &force);
    ret = sk;
  } else { 
    /* find appropriate subflow for its usage */
    ret = get_redundant_subflow(mpcb, skb, &dummy_selector, zero_wnd_test, &force);
  }

out:
  return ret;
}

/* returns the next segment to be sent from the mptcp meta-queue.
 * (chooses the reinject queue if any segment is waiting in it, otherwise,
 * chooses the normal write queue).
 * sets *@reinject to 1 if the returned segment comes from the
 * reinject queue. 
 * sets it to  0  if it is the regular send-head of the meta-sk,
 * sets it to -1  if it is a meta-level retransmission to optimize the
 * receive-buffer.
 */
static struct sk_buff *__raven_next_segment(struct sock *meta_sk, int *reinject, int *redundant)
{
  struct sock *sk;
  struct tcp_sock *meta_tp = tcp_sk(meta_sk);
	struct mptcp_cb *mpcb = meta_tp->mpcb;
	struct sk_buff *skb = NULL;
  int cnt_redundant = -1; 
  u8 pm = 0;
  bool unused_found = false;
  bool from_rdn_wqueue = false;

	/* if we are in fallback-mode, just take from the meta-send-queue */
	if (mpcb->infinite_mapping_snd || mpcb->send_infinite_mapping)
		return tcp_send_head(meta_sk);

  /* pop from mpcb's reinject_queue */
  if ( (skb = skb_peek(&mpcb->reinject_queue)) ) {
    u32 seq, eseq;
    seq = TCP_SKB_CB(skb)->seq;
    eseq = TCP_SKB_CB(skb)->end_seq;

    /* MICRO: benefit of Raven in kernel */
    /* if (before(meta_tp->snd_una, TCP_SKB_CB(skb)->seq) && meta_tp->snd_una != seq)  */
    if (meta_tp->snd_una > TCP_SKB_CB(skb)->seq &&
        seq <= meta_tp->redundant_snd_nxt &&seq <= meta_tp->snd_nxt) 
    {
#if 1
      /* pr_emerg("[%s] raven seq %u freed, snd_una %u, snd_nxt %u, " */
      /*     "snd_rdn_nxt %u, rdn? %s\n", __func__, seq, meta_tp->snd_una, */
      /*     meta_tp->snd_nxt, meta_tp->redundant_snd_nxt, */ 
      /*     (mptcp_is_skb_redundant(skb) ? "T" : "F")); */
      __skb_unlink(skb, &mpcb->reinject_queue);
      kfree_skb(skb);
      skb = NULL;
      goto out;
#endif
    }

#if 1
    /* pr_crit("[%s] from rqueue: rdn %s, " */
    /*   "seq %u, len %u, pm %u, rpm %u, " */
    /*   "meta [snd_una %u, snt_nxt %u, redundant_snd_nxt %u]\n" */
    /*   , __func__, (mptcp_is_skb_redundant(skb) ? "T" : "F"), */
    /*   TCP_SKB_CB(skb)->seq, skb->len, */ 
    /*   TCP_SKB_CB(skb)->path_mask, skb->redundant_path_mask, */ 
    /*   meta_tp->snd_una, meta_tp->snd_nxt, meta_tp->redundant_snd_nxt); */
    if (mptcp_is_skb_redundant(skb)) {
      __skb_unlink(skb, &mpcb->reinject_queue);
      kfree_skb(skb);
      goto check_rwqueue;
    }

#if 0 
  if (mpcb->raven_policy->redundancy && sysctl_mptcp_raven_measure) 
  {
    if (!skb_queue_empty(&mpcb->raven_write_queue)) 
    {
      if ( (tmp_skb = skb_peek(&mpcb->raven_write_queue)) ) 
      {
        if (TCP_SKB_CB(tmp_skb)->seq < TCP_SKB_CB(skb)->seq)
          goto check_rwqueue;
      }
    }
  }
#endif

#endif

    TCP_SKB_CB(skb)->mptcp_flags |= MPTCP_REINJECT;
    *reinject = 1;
    goto out;
  } 

check_rwqueue:
  /* acquire redundant_lock to ensure no race-condition. 
   * skb_peek returns *head* of redundant_queue. 
   * this is why we add at a tail of queue from enqueue operation.
   */
  if ( (skb = skb_peek(&mpcb->raven_write_queue)) ) {
    sk = get_valid_next_subflow(skb, mpcb);
    unused_found = false;

    if (!sk) 
      goto check_meta_wqueue;

    /* reset to initial slow start */
    if (tcp_sk(sk)->snd_cwnd < 10) 
      tcp_sk(sk)->snd_cwnd = 10;

/* check_again: */
    if (!raven_mptcp_is_temp_unavailable(sk, skb, false))
      unused_found = true;

    if (!unused_found) {
      /* pr_emerg("[%s] forcing meta wqueue to check...! bc/ %u\n", */ 
      /*     __func__, mptcp_path_index(sk)); */
      goto check_meta_wqueue;
    }

    skb->cnt_redundant--;
    cnt_redundant = skb->cnt_redundant;
    *redundant = cnt_redundant;
    from_rdn_wqueue = true;
    pm = skb->redundant_path_mask;

    if (!cnt_redundant && unused_found) {
      /* been sent over all subflows. move to rtx queue */
      __skb_unlink(skb, &mpcb->raven_write_queue);

      /* put dequeued rdn_skb into redundant_rtx_queue just 
       * in a case all rdn pkts over all subflows are lost.
       */
      skb_queue_tail(&mpcb->raven_rtx_queue, skb);

      /* rdn_skb been sent all subflows,
       * tell meta_tp that ok to send next seg
       */
      /* if (!skb->redundant_reinjected) */
        meta_tp->snd_nxt = TCP_SKB_CB(skb)->end_seq;
    }
    BUG_ON(cnt_redundant < -1);
  } else {
    int cnt = 1;
check_meta_wqueue:
again:
    /* dequeueing from meta_sk's write_queue */
		skb = tcp_send_head(meta_sk);
    if (!skb) 
      *reinject = 0;
    else if (mptcp_is_skb_redundant(skb)) 
    {
      /* if dequeued from meta-sk, it should be fresh, but b/c skb is
       * often recycled, we need to unmask mptcp_redundant flag
       */
      /* mptcp_debug("removing redundant flag %u %u\n", TCP_SKB_CB(skb)->seq, */
      /*     TCP_SKB_CB(skb)->end_seq); */

      TCP_SKB_CB(skb)->mptcp_flags &= ~MPTCP_REDUNDANT;

      if (tcp_skb_is_last(meta_sk, skb))
        meta_sk->sk_send_head = NULL;
      else
        meta_sk->sk_send_head = tcp_write_queue_next(meta_sk, skb);
      cnt--;
      if (cnt > 0)
        goto again;
    }
	}

out:
	return skb;
}

static struct sk_buff *raven_next_segment(struct sock *meta_sk, 
    int *reinject, struct sock **subsk, unsigned int *limit)
{
	struct sk_buff *skb;
	struct tcp_sock *subtp;
  unsigned int mss_now;
	u16 gso_max_segs;
	u32 max_len, max_segs, window, needed;
  int redundant = 0;
  bool returnnull = false;
  struct tcp_sock *meta_tp = tcp_sk(meta_sk);
	struct mptcp_cb *mpcb = meta_tp->mpcb;

  *reinject = 0;

	/* as we set it, we have to reset it as well. */
	*limit = 0;

  /* /1* if (sysctl_mptcp_raven_measure && mpcb && (mpcb->raven_policy->redundancy)) { *1/ */
  /*     pr_emerg("[%s] qlen %u, %u, %u, snd_una %u, snd_nxt %u\n", */
  /*         __func__, */ 
  /*         skb_queue_len(&((meta_sk)->sk_write_queue)), */
  /*         skb_queue_len(&mpcb->raven_write_queue), */
  /*         skb_queue_len(&mpcb->raven_rtx_queue), */
  /*         meta_tp->snd_una, */
  /*         meta_tp->snd_nxt */
  /*         ); */
  /*   if (tcp_send_head(meta_sk)) */
  /*     pr_emerg("seq %u\n", TCP_SKB_CB(tcp_send_head(meta_sk))->seq); */
  /*   if (!skb_queue_empty(&((meta_sk)->sk_write_queue))) */
  /*       pr_emerg("meta %u\n", TCP_SKB_CB(tcp_write_queue_head(meta_sk))->seq); */
  /* /1* } *1/ */

	if (sysctl_mptcp_raven_switching) { 
		if (mpcb && mpcb->stripe_mode) { 
			if (!skb_queue_len(&((meta_sk)->sk_write_queue))) {
				mpcb->stripe_mode = false;
				pr_emerg("[%s] switching to redundancy! sb ?, qlen 0\n", __func__);
				to_redundancy += 1;
			}
		}
  }

  /* for other policies, use default next_segment note that for principled
   * redundancy segments, it will return redundant int set 
   */
  skb = __raven_next_segment(meta_sk, reinject, &redundant); 
	if (!skb) {
    /* /1* if (sysctl_mptcp_raven_measure) *1/ */ 
    /*   pr_emerg("[%s] returning NULL! (no more data - %u, %u, %u)\n", __func__, */
    /*       skb_queue_len(&((meta_sk)->sk_write_queue)), */
    /*       skb_queue_len(&mpcb->raven_write_queue), */
    /*       skb_queue_len(&mpcb->raven_rtx_queue) */
    /*       ); */
		return NULL;
  }

  /* redundant : int
   *  2 - unlink this skb from the queue as this subsk is the last subflow that
   *      sends redundant data.
   *  x - there is x - 2 more subflows to send over.
   */
	*subsk = raven_get_subflow(meta_sk, skb, false);
	if (!*subsk) {
    /* /1* if (sysctl_mptcp_raven_measure) *1/ */ 
    /*   pr_emerg("[%s] returning NULL! (no subflow)\n", __func__); */
		return NULL;
  }

#if 0 // FIXME: later...
  if (mpcb->raven_policy->redundancy && sysctl_mptcp_raven_collect_samples) 
  {
    if (!tcp_sk(*subsk)->mptcp->lambda) {
      struct raven_sock_data *rsd = raven_get_data(tcp_sk(*subsk));
      int pi_idx = 0;
      if (tcp_sk(*subsk)->is_wifi) { 
        pi_idx = 2;
      } else {
        if (mptcp_path_index(*subsk) == 1)
          pi_idx = 0;
        else
          pi_idx = 1;
      }
      rsd->pi_idx = pi_idx;
      tcp_sk(*subsk)->mptcp->lambda = lambdas[pi_idx];
      pr_emerg("pi %u, is wifi? %s, lambda %d\n", mptcp_path_index(*subsk),
          (tcp_sk(*subsk)->is_wifi ? "T" : "F"), lambdas[pi_idx]
          );
    }
  }
#endif


  if (skb && skb->len > 1428) {
    pr_emerg("[%s] pi %u, seq %u, len %u,\n",__func__,
          mptcp_path_index(*subsk),
          TCP_SKB_CB(skb)->seq, skb->len);
    returnnull = true;
    goto out;
  }

  redundant = skb->cnt_redundant;
  /* if (likely(mpcb->raven_policy->redundancy) && */
  if (redundant >=0 &&
      likely(mptcp_is_skb_redundant(skb)) && !raven_redundant_pi(skb, tcp_sk(*subsk)))
  {
    if (!redundant) {
      TCP_SKB_CB(skb)->mptcp_flags |= MPTCP_REINJECT;
    }
    *reinject = redundant + 2;
#if 0
      u32 seq, eseq;
      seq = TCP_SKB_CB(skb)->seq;
      eseq = TCP_SKB_CB(skb)->end_seq;
      pr_emerg("sending redundantly pi %u, len %u, seq %u, "
          "snd_nxt %u, cnt_red %u, qlen %u, rdn_pm %u, rj %u\n",
          tcp_sk(*subsk)->mptcp->path_index, skb->len,
          /* seq, eseq, meta_tp->snd_nxt - mpcb->rdn_init_seq, redundant, */
          seq, meta_tp->snd_nxt, redundant,
          skb_queue_len(&((meta_sk)->sk_write_queue)),
          skb->redundant_path_mask,
          *reinject
        );

    if (*reinject == 3) { 
      pr_emerg("seq %u, qlen %u, last? %s\n",
          TCP_SKB_CB(skb)->seq,
          skb_queue_len(&((meta_sk)->sk_write_queue)),
          (tcp_skb_is_last(meta_sk, skb) ? "T" : "F"));
    }

#endif
    goto out;
  }

	subtp = tcp_sk(*subsk);
	mss_now = tcp_current_mss(*subsk);

  /* no splitting required, as we will only send one single segment */
	if (skb->len <= mss_now)
    goto out;

	/* the following is similar to tcp_mss_split_point, but
	 * we do not care about nagle, because we will anyways
	 * use tcp_nagle_push, which overrides this.
	 *
	 * so, we first limit according to the cwnd/gso-size and then according
	 * to the subflow's window.
	 */
	gso_max_segs = (*subsk)->sk_gso_max_segs;
	if (!gso_max_segs) /* no gso supported on the subflow's nic */
		gso_max_segs = 1;

	max_segs = min_t(unsigned int, mptcp_cwnd_test(subtp, skb), gso_max_segs);
	if (!max_segs)
		return NULL;

	max_len = mss_now * max_segs;
	window = tcp_wnd_end(subtp) - subtp->write_seq;

	needed = min(skb->len, window);
	if (max_len <= skb->len)
		/* take max_win, which is actually the cwnd/gso-size */
		*limit = max_len;
	else
		/* or, take the window */
		*limit = needed;

out:
  if (likely(skb && *subsk)) {
    u32 seq, eseq;
    seq = TCP_SKB_CB(skb)->seq;
    eseq = TCP_SKB_CB(skb)->end_seq;
#if 1
    /* /1* if (sysctl_mptcp_raven_measure) *1/ */ 
    /* pr_crit("[%s] pi %u, cnt_redudnant %d, " */
    /*     "seq %u, eseq %u, pm %u, len %u, " */
    /*     "meta [snd_una %u, snt_nxt %u, redundant_snd_nxt %u] rj %u, wd %s\n" */
    /*     , __func__,  mptcp_path_index(*subsk), */
    /*     skb->cnt_redudnant, seq, eseq, TCP_SKB_CB(skb)->path_mask, skb->len, */
    /*     meta_tp->snd_una, meta_tp->snd_nxt, meta_tp->redundant_snd_nxt, */
    /*     *reinject, */
    /*     !after(TCP_SKB_CB(skb)->end_seq, tcp_wnd_end(meta_tp)) ? "T" : "F" */
    /*     ); */
#endif
    if (returnnull)
      skb = NULL;
  }

	return skb;
}

/* does at least the first segment of skb fit into the send window? */
bool mptcp_snd_wnd_test(const struct tcp_sock *tp, const struct sk_buff *skb,
		      unsigned int cur_mss)
{
  // get meta-tp's end_seq
	u32 end_seq = TCP_SKB_CB(skb)->end_seq;

	if (skb->len > cur_mss)
		end_seq = TCP_SKB_CB(skb)->seq + cur_mss;

  // tcp_wnd_end: retrun tp->snd_una + tp->snd_wnd
	return !after(end_seq, tcp_wnd_end(tp));
}

/* can at least one segment of skb be sent right now, according to the
 * congestion window rules?  if so, return how many segments are allowed.
 */
unsigned int mptcp_cwnd_test(const struct tcp_sock *tp,
			   const struct sk_buff *skb)
{
	u32 in_flight, cwnd;

	/* don't be strict about the congestion window for the final fin.  */
  if (skb && (TCP_SKB_CB(skb)->tcp_flags & TCPHDR_FIN) &&
	    tcp_skb_pcount(skb) == 1)
		return 1;

	in_flight = tcp_packets_in_flight(tp);
	cwnd = tp->snd_cwnd;
	if (in_flight < cwnd)
		return (cwnd - in_flight);
	return 0;
}

/* note: make sure to `raven_close' is invoked from **server** side 
 * for redundancy policy. this is ducttape-fix but seems to be working.
 * why? client calls `mptcp_sub_close_passive' from mp_fin pkt and closes
 * the connection. in reverse case, server is stuck in dead-lock or ignores
 * sub_close_passive from `rcv_state_process' in tcp functions.
 */
static void raven_release(struct sock *sk) 
{
  struct tcp_sock *tp = tcp_sk(sk);
  struct sk_buff_head *raven_rtx_queue;
  struct sk_buff *skb, *tmp;

  /* whichever that frees up late */
  if (tp->mpcb->cnt_established == 1) {
    skb_queue_purge(&tp->mpcb->raven_write_queue);
    raven_rtx_queue = &tp->mpcb->raven_rtx_queue;
    skb_queue_walk_safe(raven_rtx_queue, skb, tmp) {
      if (skb) 
        kfree_skb(skb);
    }
    skb_queue_purge(&tp->mpcb->raven_storage_queue);
  }
  return;
}

/* sched_ops init */
static void raven_sched_init(struct sock *sk)
{
  struct tcp_sock *meta_tp, *tp = tcp_sk(sk);
  struct mptcp_cb *mpcb;
  struct raven_sock_data *rsd = raven_get_data(tp);
  int pi_idx = 0;

  rsd->pi_idx = -1;

  has_been_est = false;
  has_been_sent = 0;

  if (sk && tcp_sk(sk) && tcp_sk(sk)->mptcp) {
    meta_tp = mptcp_meta_tp(tcp_sk(sk));
    mpcb = tcp_sk(sk)->mpcb;
    mpcb->cnt_rcv_redundant_skb = 0;
  }

  mpcb = tcp_sk(sk)->mpcb;

  if (likely(tp && tp->mptcp)) {
    tp->mptcp->lambda = 0;
  }

#if 0
#if 0
  if (tp && tp->mptcp) {
    bool saw_140 = false;
    bool saw_150 = false;
    if (mptcp_path_index(sk) == 3) {
      struct sock *tmpsk;
      mptcp_for_each_sk(mpcb, tmpsk) {
        struct inet_sock *isk = inet_sk(tmpsk);
        char source[16];
        if (iamserver) 
          snprintf(source, 16, "%pI4", &(isk->inet_daddr));
        else
          snprintf(source, 16, "%pI4", &(isk->inet_saddr));

        if (!strcmp(source, "192.168.140.3")) {
          saw_140 = true;
          tcp_sk(tmpsk)->mptcp->lambda = lambdas[1];
        }

        if (!strcmp(source, "192.168.150.3")) {
          saw_150 = true;
          tcp_sk(tmpsk)->mptcp->lambda = lambdas[2];
        }

        pr_emerg("source %s, saw 140? %s, saw 150? %s, cnt_subflow %u, "
            "lambda %u\n", source, (saw_140 ? "T" : "F"),
            (saw_150 ?  "T" : "F"), mpcb->cnt_subflows,
            tcp_sk(tmpsk)->mptcp->lambda);
      }
    }
#endif

#if defined(CASE) && (CASE == 1 || CASE == 4)
    if (mptcp_path_index(sk) == 1) {
      pi_idx = 0;
    } else if (mptcp_path_index(sk) == 3) {
      if (saw_140) 
        pi_idx = 2;
      else if (saw_150)
        pi_idx = 1;
    }
#elif defined(CASE) && (CASE == 2 || CASE == 3)
    /* for 2 subflow case, pi=2 is always sprint */
    if (mptcp_path_index(sk) == 1) {
      pi_idx = 0;
    } else if (mptcp_path_index(sk) == 2) {
      pi_idx = 1;
    }
#endif
    tcp_sk(sk)->mptcp->lambda = lambdas[pi_idx];
  }
#endif

  pr_emerg("[%s] RAVEN: pi=%u lambda idx=%d for tracce=%d, CI=%d\n", __func__,
			mptcp_path_index(sk), pi_idx, target_case, target_ci);
}

static struct mptcp_sched_ops mptcp_sched_raven = {
  .get_subflow = raven_get_subflow,
  .next_segment = raven_next_segment,
  .release = raven_release,
  .init = raven_sched_init,
  .name = "raven",
  .owner = THIS_MODULE,
};

static int __init raven_register(void) 
{
	BUILD_BUG_ON(sizeof(struct raven_sock_data) > MPTCP_SCHED_SIZE);

	if (mptcp_register_scheduler(&mptcp_sched_raven)) 
    goto fail;

	pr_emerg("[%s] RAVEN: trace=%d, confidence interval=%d\n", __func__,
			target_case, target_ci);

	if (target_case < 1 || target_case > 4) 
	{
		pr_emerg("[%s] RAVEN supports only target case 1-4 for now. "
				" Using target 2 case as a default\n", __func__);
		target_case = 2;
	} 

	switch(target_case) {
		case 1:
			lambdas = lambdas1;
			break;
		case 2:
			lambdas = lambdas2;
			break;
		case 3:
			lambdas = lambdas3;
			break;
		case 4:
			lambdas = lambdas4;
			break;
	}

  return 0;
fail:
  return -1;
}

static void __exit raven_unregister(void)
{
	mptcp_unregister_scheduler(&mptcp_sched_raven);
}

module_init(raven_register);
module_exit(raven_unregister);

module_param(target_case, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
MODULE_PARM_DESC(target_case, "replay trace");
module_param(target_ci, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
MODULE_PARM_DESC(target_ci, "confidence interval");

MODULE_AUTHOR("Hyunjong Joseph Lee");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("MPTCP RAVEN scheduler");
MODULE_VERSION("0.93");
