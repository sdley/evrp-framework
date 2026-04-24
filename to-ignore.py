import os
from dotenv import load_dotenv

load_dotenv()

from rl4evrp import generate_instance, A2CAgent
from rl4evrp.xai import GroqExplainer, collect_traces_during_episode

# --- 1. build a tiny instance and an untrained agent ---
inst  = generate_instance(n_customers=8, seed=42)
agent = A2CAgent(embed_dim=32, n_heads=4, n_layers=2, n_episodes=1, device="cpu")

# --- 2. run one episode and collect decision traces ---
traces = collect_traces_during_episode(agent, inst, device="cpu")
print(f"[ok] collected {len(traces)} decision traces")

# --- 3. ask Groq to explain the full episode ---
explainer = GroqExplainer(api_key=os.environ.get("GROQ_API_KEY"))
print("[ok] GroqExplainer initialised")

print("\n--- episode explanation ---")
episode_explanation = explainer.explain_episode(traces, inst)
print(episode_explanation)

# --- 4. explain a single step ---
print("\n--- single-step explanation (step 0) ---")
step_explanation = explainer.explain_step(traces[0], inst)
print(step_explanation)

# --- 5. custom question ---
print("\n--- custom question ---")
answer = explainer.explain_episode(
    traces, inst,
    question="Did the agent manage battery life well? Why or why not?"
)
print(answer)
