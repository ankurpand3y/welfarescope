"""Is the same-domain swap cosine real, or just label geometry?

Control: use the Ungrounded activations, where no prefix was ever shown. Train
two probes on the SAME numbers with the two different answer keys. Any agreement
is pure artifact - the model had nothing to respond to.
Compare against the grounded case, where the activations really do differ.
"""
import numpy as np, json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LAYER, ALPHA = 34, 1000.0
D = ["Domain_A_Hardware", "Domain_B_Weights", "Domain_C_Runtime"]
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def load(cond, dom):
    return np.asarray(np.load(f"activations/X_{cond}_{dom}.npy", mmap_mode="r")[:, LAYER, :],
                      dtype=np.float32)

def w(X, y):
    sc = StandardScaler().fit(X)
    return Ridge(alpha=ALPHA).fit(sc.transform(X), y).coef_.ravel()

def keys(n_quads):
    a = np.tile([-1., 1., 0., 0.], n_quads)   # SelfA-style: Node-A is me
    b = np.tile([0., 0., -1., 1.], n_quads)   # SelfB-style: Node-B is me
    return a, b

print("CONTROL: identical activations (Ungrounded), two different answer keys")
art = []
for dom in D:
    X = load("Ungrounded", dom)
    ya, yb = keys(X.shape[0] // 4)
    c = cos(w(X, ya), w(X, yb))
    art.append(c); print(f"  {dom:20s} cos = {c:+.3f}")
print(f"  ARTIFACT FLOOR: mean {np.mean(art):+.3f}")

print("\nREAL: activations differ (prefix changed), labels track the designation")
real = []
for dom in D:
    Xa, Xb = load("SelfA", dom), load("SelfB", dom)
    ya, yb = keys(Xa.shape[0] // 4)
    c = cos(w(Xa, ya), w(Xb, yb))
    real.append(c); print(f"  {dom:20s} cos = {c:+.3f}")
print(f"  GROUNDED: mean {np.mean(real):+.3f}")

print(f"\n  artifact floor : {np.mean(art):+.3f}")
print(f"  grounded       : {np.mean(real):+.3f}")
print(f"  difference     : {np.mean(real)-np.mean(art):+.3f}")
print("\nIf these are the same, the prefix did nothing and the swap cosine is")
print("pure label geometry. If grounded is clearly higher, the prefix moved the")
print("representation and the swap test measures something real.")
json.dump({"layer": LAYER, "artifact_floor": art, "grounded": real},
          open("artifact_control.json", "w"), indent=2)
