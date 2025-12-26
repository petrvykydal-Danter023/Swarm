import jax
import jax.numpy as jnp
from functools import partial
from entropy.core.world import WorldState
from entropy.config import IntentConfig

@partial(jax.jit, static_argnums=(2,))
def process_intent(
    state: WorldState,
    actions: jnp.ndarray,
    config: IntentConfig
) -> jnp.ndarray:
    """
    Překládá High-Level Intents na Low-Level Motor Commands.
    
    Pokud config.enabled == False, vrací akce beze změny (Direct Control).
    
    Intents (Action Space):
    [0]: Intent Type (0=Velocity, 1=TargetRel)
    [1]: Param 1 (Speed / RelX)
    [2]: Param 2 (Rotation / RelY)
    [3:]: Ostatní (Gate, Comm...) - kopírují se
    
    Output (Motor Action Space):
    [0]: Motor L [-1, 1]
    [1]: Motor R [-1, 1]
    [2:]: Ostatní
    """
    if not config.enabled:
        return actions
        
    N = state.agent_positions.shape[0]
    
    # Rozparsuj intent
    # Softmax/Argmax pro typ by byl ideální, ale pro PPO continuous space
    # použijeme práh: < 0.0 -> Velocity, > 0.0 -> Target
    intent_type_logit = actions[:, 0]
    is_target_mode = intent_type_logit > 0.0
    
    param1 = actions[:, 1]
    param2 = actions[:, 2]
    rest_actions = actions[:, 3:]
    
    # === REŽIM A: VELOCITY CONTROL (v, omega) ===
    # param1 = linear_velocity [-1, 1]
    # param2 = angular_velocity [-1, 1]
    # Unicycle model -> Differential Drive
    # L = v - omega
    # R = v + omega
    vel_cmd_L = jnp.clip(param1 - param2, -1.0, 1.0)
    vel_cmd_R = jnp.clip(param1 + param2, -1.0, 1.0)
    
    velocity_motor_cmds = jnp.stack([vel_cmd_L, vel_cmd_R], axis=1)
    
    # === REŽIM B: TARGET CONTROL (RelX, RelY) ===
    # param1 = RelX (vzdálenost dopředu/dozadu)
    # param2 = RelY (vzdálenost vlevo/vpravo)
    # Cíl je relativně k agentovi
    
    # PID logika pro natočení k cíli a jízdu k němu
    # Target Angle (kde je cíl relativně k pohledu agenta)
    target_angle = jnp.arctan2(param2, param1) # [-pi, pi]
    target_dist = jnp.sqrt(param1**2 + param2**2)
    
    # Simple P-Controller
    # Pokud je úhel velký, toč se na místě. Pokud malý, jeď dopředu.
    
    # Angular error (chceme target_angle = 0)
    ang_err = target_angle
    
    # Linear error (chceme dojít tam)
    # Pokud jsme zády k cíli, necouvejme (nebo ano? pro teď raději otočka)
    # Všimni si: atan2 vrací úhel k bodu. 
    
    # Control logic:
    # Turn = P * ang_err
    # Drive = P * dist * cos(ang_err) (aby nejel full speed když se točí)
    
    turn_cmd = jnp.clip(ang_err * config.pid_rot_kp, -1.0, 1.0)
    
    # Zpomal pokud musíš ostře zatáčet
    forward_damp = jnp.maximum(0.0, jnp.cos(ang_err)) 
    drive_cmd = jnp.clip(target_dist * config.pid_pos_kp, -1.0, 1.0) * forward_damp
    
    tgt_cmd_L = jnp.clip(drive_cmd - turn_cmd, -1.0, 1.0)
    tgt_cmd_R = jnp.clip(drive_cmd + turn_cmd, -1.0, 1.0)
    
    target_motor_cmds = jnp.stack([tgt_cmd_L, tgt_cmd_R], axis=1)
    
    # === SELECT MODE ===
    # Vectorized selection based on is_target_mode mask
    # [N, 2]
    final_motors = jnp.where(
        is_target_mode[:, None],
        target_motor_cmds,
        velocity_motor_cmds
    )
    
    # Slep zpátky s ostatními akcemi (Gate, Comm...)
    # Pozor: Input actions měly [IntentType, P1, P2, ...Rest]
    # Output actions mají [MotL, MotR, ...Rest] (IntentType zmizel)
    # Ale env_wrapper očekává stejnou dimenzi? Ne, env_wrapper očekává MOTOR control.
    # Pokud PPO outputuje Intents, env_wrapper musí vědět, že prvních X dimenzí jsou Intenty.
    # Ale `step` funkce v env_wrapper bere `actions`.
    # A `physics_step` bere `actions[:, :2]`.
    # Takže pokud nahradíme první 2 sloupce motory, bude to fungovat!
    # A co Gate? Ten je na indexu 2.
    # V Intentech je Gate na indexu 3 (protože 0=Type, 1=P1, 2=P2).
    # Musíme tedy posunout Rest o 1 doleva nebo jinak namapovat.
    
    # PROBLÉM: Změna dimenze akcí. 
    # Direct: [MotL, MotR, Gate, ...]
    # Intent: [Type, P1, P2, Gate, ...] -> o 1 víc.
    
    # Řešení: Předpokládejme, že Intent mód má ActionDim + 1.
    # A výstup z `process_intent` bude [MotL, MotR, Gate, ...] (ActionDim - 1 vzhledem k inputu, resp. standardní dimenze enginu)
    
    return jnp.concatenate([final_motors, rest_actions], axis=1)
