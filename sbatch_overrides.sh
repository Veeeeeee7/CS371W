# =============================================================================
# sbatch_overrides.sh -- shared sbatch CLI override surface for the fan-outs
#
# =============================================================================
# SOURCED, not executed. submit_aotil_all.sh sources this and passes
# "${SB_OVERRIDES[@]}" to sbatch. sbatch command-line options BEAT the in-script
# #SBATCH directives, so this is how a fan-out retargets the driver without
# editing the driver's header.
#
# Knobs (all env, all optional except ACCOUNT, which defaults):
#   ACCOUNT    charging account. DEFAULT general. On this cluster priority is
#              1000*Age + 10000*FairShare + 1000*QOS, and `ai-gpu`'s fairshare
#              sits far below `general`'s -- the difference is worth two orders
#              of magnitude in priority points. ACCOUNT=ai-gpu to go back.
#   QOS        --qos. Unset by default; only `normal` is normally held, so this
#              is a no-op unless the admins grant something else.
#   PARTITION  --partition
#   CPUS       --cpus-per-task
#   MEM        --mem
#   GPUS       --gpus
#   WALLTIME   --time
# Anything unset is not emitted, so the driver's own #SBATCH value stands.
#
# PriorityWeightJobSize is 0 here, so CPUS/MEM/WALLTIME do NOT change where a
# job ranks in the queue. They matter for backfill fit and for how fast the
# account's fairshare drains. Never shrink WALLTIME below a job's real runtime:
# Slurm kills at the limit and a killed cell leaves no .npz at all.
# =============================================================================

# Every driver's #SBATCH --output/--error points at logs/sbatch/%x_%j.{out,err}.
# Slurm does NOT create intermediate directories, so a missing logs/sbatch/ fails
# the job instantly with "Unable to open file". Guarantee it here.
mkdir -p logs/sbatch

ACCOUNT="${ACCOUNT:-general}"

SB_OVERRIDES=(--account "$ACCOUNT")
if [ -n "${QOS:-}" ];       then SB_OVERRIDES+=(--qos "$QOS"); fi
if [ -n "${PARTITION:-}" ]; then SB_OVERRIDES+=(--partition "$PARTITION"); fi
if [ -n "${CPUS:-}" ];      then SB_OVERRIDES+=(--cpus-per-task "$CPUS"); fi
if [ -n "${MEM:-}" ];       then SB_OVERRIDES+=(--mem "$MEM"); fi
if [ -n "${GPUS:-}" ];      then SB_OVERRIDES+=(--gpus "$GPUS"); fi
if [ -n "${WALLTIME:-}" ];  then SB_OVERRIDES+=(--time "$WALLTIME"); fi

sb_overrides_banner() {   # $1 = tag for the log line
    echo "[${1:-fan-out}] account=$ACCOUNT${QOS:+ qos=$QOS}${PARTITION:+ partition=$PARTITION}" \
         "${CPUS:+cpus=$CPUS}${MEM:+ mem=$MEM}${GPUS:+ gpus=$GPUS}${WALLTIME:+ time=$WALLTIME}"
}
