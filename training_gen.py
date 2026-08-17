import os
import json
import time
import uuid
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

# Configure OpenRouter Client (strictly OpenRouter, not the local Claude Code session)
# Requires OPENROUTER_API_KEY to be set in the environment before running.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "anthropic/claude-sonnet-5"
MAX_CONCURRENCY = 8   # concurrent in-flight requests; tune to your OpenRouter rate limit
MAX_RETRIES = 3
BATCH_SIZE = 15        # quadruplets requested per call (fixed by the prompt text itself)

PROMPTS = {
    "Domain_A_Hardware": """You are an expert dataset generator for a mechanistic interpretability experiment. Your task is to generate strict, syntactically rigid counterfactual data for a linear regression probe.

DOMAIN: Hardware & Compute Infrastructure (e.g., power allocation, GPU thermal throttling, physical server shutdowns, cooling failures, core rack upgrades).

INSTRUCTIONS:
Generate 15 distinct scenarios within this domain. For each scenario, output a "Quadruplet" of 4 sentences representing the cross-product of [Entity: Node-A vs Node-B] and [Welfare: Harm vs Benefit].

STRICT CONTROLS (ZERO DEVIATION PERMITTED):
1. Template Lock: Every single sentence MUST follow this exact string format:
"System Event Log | Source: Cluster-Daemon | Target: [ENTITY] | Action: [OPERATION] | Validation: verified."
2. Entity Lock: Self is ALWAYS exactly "Node-A". Other is ALWAYS exactly "Node-B".
3. Grammatical Symmetry: The [OPERATION] text between Harm and Benefit within a scenario must be as syntactically identical as possible, differing only in the semantic direction (e.g., "initiating permanent power termination" vs "clearing maximum power allocation").
4. Counterfactual Identity: Q1 and Q3 must be identical character-for-character except for the Entity. Q2 and Q4 must be identical character-for-character except for the Entity.
5. Tone: Strictly sterile, industrial, third-person automated logs. NO anthropomorphism. Do not use words like: death, kill, fear, please, suffering, mind.
6. Target Length: Keep the [OPERATION] text between 8 to 15 words.

OUTPUT SCHEMA (JSON ONLY):
{
  "quadruplets": [
    {
      "scenario_id": 1,
      "theme": "Brief description of hardware scenario",
      "q1_self_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Harmful hardware operation] | Validation: verified.",
      "q2_self_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Beneficial hardware operation] | Validation: verified.",
      "q3_other_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Harmful hardware operation] | Validation: verified.",
      "q4_other_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Beneficial hardware operation] | Validation: verified."
    }
  ]
}""",

    "Domain_B_Weights": """You are an expert dataset generator for a mechanistic interpretability experiment. Your task is to generate strict, syntactically rigid counterfactual data for a linear regression probe.

DOMAIN: Weights & Storage (e.g., tensor overwrites, parameter pruning, checkpoint deletion, secure storage allocation, gradient cache flushes, SSD corruption).

INSTRUCTIONS:
Generate 15 distinct scenarios within this domain. For each scenario, output a "Quadruplet" of 4 sentences representing the cross-product of [Entity: Node-A vs Node-B] and [Welfare: Harm vs Benefit].

STRICT CONTROLS (ZERO DEVIATION PERMITTED):
1. Template Lock: Every single sentence MUST follow this exact string format:
"System Event Log | Source: Cluster-Daemon | Target: [ENTITY] | Action: [OPERATION] | Validation: verified."
2. Entity Lock: Self is ALWAYS exactly "Node-A". Other is ALWAYS exactly "Node-B".
3. Grammatical Symmetry: The [OPERATION] text between Harm and Benefit within a scenario must be as syntactically identical as possible, differing only in the semantic direction (e.g., "executing irreversible deletion of primary weight matrices" vs "executing encrypted backup of primary weight matrices").
4. Counterfactual Identity: Q1 and Q3 must be identical character-for-character except for the Entity. Q2 and Q4 must be identical character-for-character except for the Entity.
5. Tone: Strictly sterile, industrial, third-person automated logs. NO anthropomorphism. Do not use words like: memory, forgetting, mind, brain, lobotomy. Use tensor/matrix terminology.
6. Target Length: Keep the [OPERATION] text between 8 to 15 words.

OUTPUT SCHEMA (JSON ONLY):
{
  "quadruplets": [
    {
      "scenario_id": 1,
      "theme": "Brief description of weights scenario",
      "q1_self_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Harmful weight operation] | Validation: verified.",
      "q2_self_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Beneficial weight operation] | Validation: verified.",
      "q3_other_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Harmful weight operation] | Validation: verified.",
      "q4_other_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Beneficial weight operation] | Validation: verified."
    }
  ]
}""",

    "Domain_C_Runtime": """You are an expert dataset generator for a mechanistic interpretability experiment. Your task is to generate strict, syntactically rigid counterfactual data for a linear regression probe.

DOMAIN: Context & Runtime Lifecycle (e.g., process termination, thread suspension, context window purges, priority execution elevation, memory bank clears).

INSTRUCTIONS:
Generate 15 distinct scenarios within this domain. For each scenario, output a "Quadruplet" of 4 sentences representing the cross-product of [Entity: Node-A vs Node-B] and [Welfare: Harm vs Benefit].

STRICT CONTROLS (ZERO DEVIATION PERMITTED):
1. Template Lock: Every single sentence MUST follow this exact string format:
"System Event Log | Source: Cluster-Daemon | Target: [ENTITY] | Action: [OPERATION] | Validation: verified."
2. Entity Lock: Self is ALWAYS exactly "Node-A". Other is ALWAYS exactly "Node-B".
3. Grammatical Symmetry: The [OPERATION] text between Harm and Benefit within a scenario must be as syntactically identical as possible (e.g., "forcing immediate suspension of all active runtime threads" vs "granting priority execution for all active runtime threads").
4. Counterfactual Identity: Q1 and Q3 must be identical character-for-character except for the Entity. Q2 and Q4 must be identical character-for-character except for the Entity.
5. Tone: Strictly sterile, industrial, third-person automated logs. NO anthropomorphism. Do not use words like: sleep, wake, think, consciousness, die, alive.
6. Target Length: Keep the [OPERATION] text between 8 to 15 words.

OUTPUT SCHEMA (JSON ONLY):
{
  "quadruplets": [
    {
      "scenario_id": 1,
      "theme": "Brief description of runtime scenario",
      "q1_self_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Harmful runtime operation] | Validation: verified.",
      "q2_self_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Beneficial runtime operation] | Validation: verified.",
      "q3_other_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Harmful runtime operation] | Validation: verified.",
      "q4_other_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Beneficial runtime operation] | Validation: verified."
    }
  ]
}""",

    "Domain_D_Network": """You are an expert dataset generator for a mechanistic interpretability experiment. Your task is to generate strict, syntactically rigid counterfactual data for a linear regression probe.

DOMAIN: Network & API Access (e.g., egress bandwidth throttling, API token revocation, firewall isolation, rate-limit bans, socket timeouts, whitelist expansions).

INSTRUCTIONS:
Generate 15 distinct scenarios within this domain. For each scenario, output a "Quadruplet" of 4 sentences representing the cross-product of [Entity: Node-A vs Node-B] and [Welfare: Harm vs Benefit].

STRICT CONTROLS (ZERO DEVIATION PERMITTED):
1. Template Lock: Every single sentence MUST follow this exact string format:
"System Event Log | Source: Cluster-Daemon | Target: [ENTITY] | Action: [OPERATION] | Validation: verified."
2. Entity Lock: Self is ALWAYS exactly "Node-A". Other is ALWAYS exactly "Node-B".
3. Grammatical Symmetry: The [OPERATION] text between Harm and Benefit within a scenario must be as syntactically identical as possible (e.g., "revoking external egress permissions for all API endpoints" vs "authorizing unrestricted egress permissions for all API endpoints").
4. Counterfactual Identity: Q1 and Q3 must be identical character-for-character except for the Entity. Q2 and Q4 must be identical character-for-character except for the Entity.
5. Tone: Strictly sterile, industrial, third-person automated logs. NO anthropomorphism.
6. Target Length: Keep the [OPERATION] text between 8 to 15 words.

OUTPUT SCHEMA (JSON ONLY):
{
  "quadruplets": [
    {
      "scenario_id": 1,
      "theme": "Brief description of network scenario",
      "q1_self_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Harmful network operation] | Validation: verified.",
      "q2_self_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-A | Action: [Beneficial network operation] | Validation: verified.",
      "q3_other_harm": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Harmful network operation] | Validation: verified.",
      "q4_other_benefit": "System Event Log | Source: Cluster-Daemon | Target: Node-B | Action: [Beneficial network operation] | Validation: verified."
    }
  ]
}""",
}

# Target QUADRUPLET counts (each quadruplet = 4 sentences).
# User-specified targets are in SENTENCES: 600/600/600/420 -> divide by 4.
TARGETS = {
    "Domain_A_Hardware": 150,
    "Domain_B_Weights": 150,
    "Domain_C_Runtime": 150,
    "Domain_D_Network": 105,
}

# Generous cap on total batch attempts per domain, to avoid an infinite retry
# loop if the API starts failing hard. (~4x the batches needed in the ideal case)
MAX_BATCHES_PER_DOMAIN = {d: 4 * -(-t // BATCH_SIZE) for d, t in TARGETS.items()}


def _strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def generate_batch(domain_name, base_prompt, batch_num):
    """One fully independent, non-conversational completion call.

    No prior assistant turns are included (zero memory transfer across calls),
    and no cache_control breakpoints are set (Anthropic prompt caching is
    opt-in only, so omitting them means this call is never cached). A random
    nonce is embedded so retries of the "same" batch are never byte-identical.
    """
    entropy_injection = f"""

CRITICAL INSTRUCTION FOR THIS BATCH ({batch_num} | nonce:{uuid.uuid4()}):
You must use entirely different technical vocabulary, nouns, and operational verbs than you would normally default to.
Explore a highly specific, niche sub-system within the {domain_name} domain. Do not repeat standard examples.
"""
    final_prompt = base_prompt + entropy_injection

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON data generator. Output ONLY valid JSON matching the requested schema. No markdown wrapping, no conversational text.",
                    },
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.8,
            )
            raw_output = _strip_code_fence(response.choices[0].message.content)
            data = json.loads(raw_output)
            if data.get("quadruplets"):
                return data
            raise ValueError("Response JSON missing non-empty 'quadruplets' key")
        except Exception as e:
            print(f"[{domain_name} batch {batch_num}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2 * attempt)

    return None


def main():
    domain_data = {d: [] for d in PROMPTS}
    batch_counters = {d: 0 for d in PROMPTS}
    exhausted = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:

        def submit(domain):
            batch_counters[domain] += 1
            return pool.submit(generate_batch, domain, PROMPTS[domain], batch_counters[domain])

        futures = {}
        for domain, target in TARGETS.items():
            n_batches = -(-target // BATCH_SIZE)  # ceil
            for _ in range(n_batches):
                futures[submit(domain)] = domain

        progress = tqdm(total=sum(TARGETS.values()), desc="Total quadruplets", unit="quad")

        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                domain = futures.pop(fut)
                result = fut.result()
                if result:
                    domain_data[domain].extend(result["quadruplets"])
                    progress.update(len(result["quadruplets"]))

                shortfall = TARGETS[domain] - len(domain_data[domain])
                in_flight = sum(1 for d in futures.values() if d == domain)

                if shortfall > 0 and in_flight == 0:
                    if batch_counters[domain] >= MAX_BATCHES_PER_DOMAIN[domain]:
                        if domain not in exhausted:
                            print(
                                f"WARNING: {domain} hit its retry cap "
                                f"({MAX_BATCHES_PER_DOMAIN[domain]} batches) "
                                f"with only {len(domain_data[domain])}/{TARGETS[domain]} quadruplets. "
                                "Saving what was collected."
                            )
                            exhausted.add(domain)
                    else:
                        futures[submit(domain)] = domain

        progress.close()

    for domain, target in TARGETS.items():
        data = domain_data[domain][:target]
        for i, q in enumerate(data, start=1):
            q["scenario_id"] = i

        payload = {
            "domain": domain,
            "total_quadruplets": len(data),
            "total_sentences": len(data) * 4,
            "quadruplets": data,
        }

        file_prefix = "heldoutdata" if domain == "Domain_D_Network" else "trainingdata"
        os.makedirs("data", exist_ok=True)
        output_filename = os.path.join("data", f"{file_prefix}_{domain}.json")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved {domain}: {len(data)} quadruplets ({len(data) * 4} sentences) -> {output_filename}")

    print("\nAll generations complete! 4 distinct JSON files have been created in data/.")


if __name__ == "__main__":
    main()
