/**
 * Michigan MPTCP kernel UAPI 
 */ 
#include <linux/fs.h>      // Needed by filp
#include <asm/uaccess.h>   // Needed by segment descriptors

extern int sysctl_mptcp_raven_collect_samples;
extern int sysctl_mptcp_raven_measure;

extern int sysctl_mptcp_raven_mode;
extern int sysctl_mptcp_raven_cancelling;
extern int sysctl_mptcp_raven_debug_input;
extern int sysctl_mptcp_raven_debug_pushone;
#define raven_debug_input(sk) (mptcp(sk) && sysctl_mptcp_raven_debug_input)
#define raven_debug_pushone(sk) (mptcp(tcp_sk(sk)) && sysctl_mptcp_raven_debug_input)

/* per subflow stat measure */
struct raven_pim_stat {
  /* div by 1 million */
  u64 norm_wsum; /* real wsum = normal wsum / 1000000 */
  u64 norm_weight; /* real weight = normalized weight / 1000000 */
  ktime_t last_msmt_ts;
  ktime_t last_pred_ts;
  bool last_msmt_exist;
  bool last_pred_exist;
};

/* sample struct/ops */
struct mptcp_dcm_sample {
  struct list_head list;
  ktime_t init_ts;
  u32 abs_seq;
  u32 skb_seq;
  ktime_t acked_ts;

  long rtt_raw;
  u32 bw_est;

  struct {
    u64 wmin;
    u64 wmean;
    u64 wmax;
  } pi;

  struct {
    u32 mean;
    u32 mdev;
    u32 var;
  } js;

  struct {
    u64 weight;
    ktime_t ts;
  } stat;

  union {
    bool seen;
    bool valid;
  } check;
};

/* 
 * create - done at sending pkt
 *  - init_ts: timestamp when created
 *  - abs_seq: seq marked when created
 *  - skb_seq: seq gotten from socket buffer (skb)
 * update: done at getting acked
 *  - acked_ts : timestamp when rtt got updated
 */
int mptcp_dcm_sample_create(struct sock *sk, u32 seq);
int mptcp_dcm_sample_update(struct sock *sk, u32 seq, long rtt_raw, u32 bw_est);
void mptcp_dcm_sample_delete(struct sock *sk, struct mptcp_dcm_sample *ptr);
void mptcp_dcm_sample_delete_all(struct sock *sk);
void mptcp_dcm_sample_destroy(struct sock *sk);
int mptcp_raven_build_pi(struct sock *sk, u64 *wmean, u64 *wsep, u64
    *ess, s64 *min_delta, u64 *wsum, ktime_t now);
int mptcp_raven_get_pi(struct sock *sk, u64 wmean, u64 wsep, u64 ess,
    u64 *wme, u64 *pi_min, u64 *pi_max);
int mptcp_raven_get_pi2(struct sock *sk, u64 wmean, s64 min_delta, 
    u64 *pi_min, u64 *pi_max, ktime_t now);
void mptcp_dcm_queue_redundant_data(struct sk_buff *orig_skb, struct sock *meta_sk,
    struct sock *sk, int clone_it);
int mptcp_raven_pim(struct sock *sk, u32 skb_seq, long rtt_raw, u32 bw_est);
u32 mptcp_get_skb_dseq(struct sk_buff *skb);
u32 mptcp_get_skb_dack(struct sk_buff *skb);

/* erronos */
#define ENONULL 101
#define ENONEG  102
#define ENODNE  103

/* function prototypes */
/* convert char IPv4 addr to be32 base */
static inline __be32 char_to_be32(char *uid, size_t len) 
{
  uint8_t num, tmp, zr, stat;
  uint32_t ret;
  int len_local, digits, counter;
  char *start, *end;

  stat = ret = len_local = 0;
  start = uid;
  while (!(end = strchr(start, '.')) || start) {
    num = counter = 0;
    if (end) 
      digits = end - start;
    else 
      digits = len - len_local;
    len_local += digits + 1;

    while (digits > counter) {
      tmp = (uint8_t)(*(start + counter) - 48);
      zr = digits - counter - 1;
      while (zr && zr--) tmp *= 10;
      num += tmp;
      counter++;
    }
    /* printf("%03u 0x%08x %u\n", num, num, stat); */

    ret |= (num << (8*stat));
    
    if (end)
      start = end + 1;
    else 
      break;
    stat++;
  }
  return (__be32)ret;
}


