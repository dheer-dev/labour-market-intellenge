from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":12,
                     "figure.dpi":120,"savefig.dpi":300,
                     "axes.spines.top":False,"axes.spines.right":False})

def load(name):
    p = DATA / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)

def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT / name)

# Figure 1: framework architecture
fig, ax = plt.subplots(figsize=(12, 6.4))
ax.set(xlim=(0,12), ylim=(0,7)); ax.axis("off")
def node(x,y,w,h,title,body,face="#F1F4F7"):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.10",
                     linewidth=1.1,edgecolor="#526273",facecolor=face)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*.68,title,ha="center",va="center",fontweight="bold",fontsize=9,wrap=True)
    ax.text(x+w/2,y+h*.32,body,ha="center",va="center",fontsize=8,wrap=True)
def arr(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,
                                 linewidth=1.1,color="#526273"))
ax.text(6,6.65,"Explainable labour-market intelligence framework",ha="center",fontsize=14,fontweight="bold")
node(.3,5.1,2.3,1,"ESCO taxonomy","Occupations, skills,\noccupation–skill relations")
node(.3,3.6,2.3,1,"NYC job postings","Posting-derived skills\nand market evidence")
node(3.2,4.25,2.2,1.25,"Data preparation","Cleaning, mapping,\nskill-demand matrix","#E7F0FA")
arr(2.6,5.55,3.2,5.05); arr(2.6,4.05,3.2,4.7)
node(6,5.1,2.3,1,"Skill-gap analysis","Compare taxonomy with\nobserved market skills","#E7F0FA")
node(6,3.6,2.3,1,"ML evaluation","TF-IDF + Logistic Regression\noccupation-aware validation","#F7EEDC")
arr(5.4,4.95,6,5.55); arr(5.4,4.75,6,4.1)
node(8.9,5.1,2.7,1,"Gap prioritisation","Prevalence-based\npriority categories","#E5F2E8")
node(8.9,3.6,2.7,1,"Evidence & explanations","Posting counts, prevalence,\nrule-based explanations","#E5F2E8")
arr(8.3,5.6,8.9,5.6); arr(8.3,4.1,8.9,4.1); arr(10.25,5.1,10.25,4.6)
node(3.8,1.15,4.3,1.2,"Career decision-support dashboard",
     "Occupation explorer • prioritised gaps\nML exploration • personalised gap view","#E7F0FA")
arr(10.25,3.6,7.4,2.35); arr(7,3.6,6.5,2.35)
ax.text(6,.5,"ML estimates observation in the available job-posting data; it does not predict future demand or universal requirements.",
        ha="center",fontsize=8,color="#444")
save(fig,"figure1_framework_architecture.png")

# Figure 2: top skills per occupation
d=load("occupation_skill_demand_analysis.csv")
req={"occupation_label","skill_label","skill_posting_count","skill_prevalence_percent"}
if not req.issubset(d.columns): raise ValueError(f"Demand columns missing: {req-set(d.columns)}")
targets=["civil engineer","construction manager","environmental engineer","lawyer","project manager"]
targets=[x for x in targets if x in set(d.occupation_label.dropna())]
if not targets: raise ValueError("Target occupations not found in demand CSV")
fig,axs=plt.subplots(len(targets),1,figsize=(10,2.1*len(targets)))
if len(targets)==1: axs=[axs]
for ax,occ in zip(axs,targets):
    z=d[d.occupation_label==occ].sort_values(["skill_posting_count","skill_prevalence_percent"],ascending=False).head(5)
    z=z.sort_values("skill_posting_count")
    bars=ax.barh(z.skill_label.astype(str).str.slice(0,42),z.skill_posting_count)
    ax.set_title(occ.title(),loc="left",fontweight="bold",fontsize=10)
    ax.set_xlabel("Relevant job postings"); ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
    for b,v in zip(bars,z.skill_posting_count):
        ax.text(b.get_width()+.3,b.get_y()+b.get_height()/2,str(int(v)),va="center",fontsize=8)
fig.suptitle("Top market-observed skills by target occupation",y=1.003,fontweight="bold",fontsize=13)
fig.tight_layout()
save(fig,"figure2_top_skills_by_occupation.png")

# Figure 3: priority distribution
c=load("career_decision_support.csv")
if not {"priority","occupation_label","skill_label"}.issubset(c.columns):
    raise ValueError("Career file needs priority, occupation_label and skill_label")
order=["High","Moderate","Emerging"]
counts=c.priority.value_counts().reindex(order,fill_value=0)
fig,ax=plt.subplots(figsize=(7.2,4.6))
bars=ax.bar(counts.index,counts.values)
ax.set(title="Distribution of prioritised skill gaps",ylabel="Number of gap records")
ax.set_ylim(0,max(counts.max()*1.22,1)); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
for b,v in zip(bars,counts.values): ax.text(b.get_x()+b.get_width()/2,v+.25,str(int(v)),ha="center",fontweight="bold")
ax.text(.5,-.18,"Categories follow the selected prevalence-based prioritisation rule.",transform=ax.transAxes,ha="center",fontsize=8)
fig.tight_layout(); save(fig,"figure3_priority_distribution.png")

# Figure 4: distinct gap records by occupation
g=c.groupby("occupation_label").skill_label.nunique().sort_values()
fig,ax=plt.subplots(figsize=(8.5,4.8))
bars=ax.barh(g.index.str.title(),g.values)
ax.set(title="Prioritised skill gaps by target occupation",xlabel="Distinct prioritised skill-gap records")
ax.set_xlim(0,max(g.max()*1.22,1)); ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
for b,v in zip(bars,g.values): ax.text(v+.12,b.get_y()+b.get_height()/2,str(int(v)),va="center",fontweight="bold")
fig.tight_layout(); save(fig,"figure4_gaps_by_occupation.png")

# Figure 5: metric comparison
s=load("nested_threshold_summary.csv")
configs=[("baseline","fixed_threshold_050","Baseline · threshold 0.50"),
         ("baseline","nested_selected_threshold","Baseline · nested threshold"),
         ("balanced","nested_selected_threshold","Balanced · nested threshold")]
metrics=["precision","recall","f1"]; display=["Precision","Recall","F1-score"]
vals=[]; labels=[]
for model,ev,label in configs:
    row=s[(s.model==model)&(s.evaluation==ev)]
    if row.empty: raise ValueError(f"Missing summary: {model}/{ev}")
    vals.append([float(row.loc[row.metric==m,"value"].iloc[0]) for m in metrics]); labels.append(label)
x=np.arange(len(labels)); w=.23
fig,ax=plt.subplots(figsize=(10,5.2))
for i,name in enumerate(display):
    bars=ax.bar(x+(i-1)*w,[v[i] for v in vals],w,label=name)
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,f"{b.get_height():.2f}",ha="center",fontsize=8)
ax.set_xticks(x,labels); ax.set_ylabel("Score"); ax.set_ylim(0,.82)
ax.set_title("Occupation-aware ML evaluation",fontweight="bold")
ax.legend(ncol=3,frameon=False); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"figure5_ml_metric_comparison.png")

# Load actual pooled outer-fold predictions
o=load("nested_threshold_oof_predictions.csv")
need={"market_observed","model","predicted_probability","selected_threshold"}
if not need.issubset(o.columns): raise ValueError(f"OOF columns missing: {need-set(o.columns)}")

# Figure 6: confusion matrices
fig,axs=plt.subplots(1,3,figsize=(12,4.2))
for ax,(model,mode,threshold) in zip(axs,[("baseline","fixed",.5),("baseline","nested",None),("balanced","nested",None)]):
    z=o[o.model==model]
    yt=z.market_observed.astype(int).to_numpy()
    pr=z.predicted_probability.astype(float).to_numpy()
    th=np.full(len(z),threshold) if threshold is not None else z.selected_threshold.to_numpy()
    yp=(pr>=th).astype(int)
    cm=confusion_matrix(yt,yp,labels=[0,1])
    ax.imshow(cm,cmap="Blues")
    title={"fixed":"Baseline · fixed 0.50","nested":"Baseline · nested threshold"}[mode]
    if model=="balanced": title="Balanced · nested threshold"
    ax.set_title(title,fontsize=9,fontweight="bold")
    ax.set_xticks([0,1],["Pred. 0","Pred. 1"]); ax.set_yticks([0,1],["Actual 0","Actual 1"])
    for (r,col),v in np.ndenumerate(cm):
        ax.text(col,r,f"{v:,}",ha="center",va="center",
                color="white" if v>cm.max()*.55 else "black",fontweight="bold")
fig.suptitle("Confusion matrices from pooled outer-fold predictions",fontweight="bold")
fig.tight_layout(); save(fig,"figure6_confusion_matrices.png")

# Figure 7: precision-recall curves
fig,ax=plt.subplots(figsize=(7.4,5.5))
prevalence=None
for model,label in [("baseline","Baseline Logistic Regression"),("balanced","Class-weighted Logistic Regression")]:
    z=o[o.model==model]
    yt=z.market_observed.astype(int).to_numpy()
    pr=z.predicted_probability.astype(float).to_numpy()
    p,r,_=precision_recall_curve(yt,pr)
    ap=average_precision_score(yt,pr)
    ax.plot(r,p,linewidth=2,label=f"{label} (AP={ap:.3f})")
    prevalence=float(yt.mean())
ax.axhline(prevalence,linestyle="--",linewidth=1,label=f"Positive prevalence ({prevalence:.3f})")
ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1))
ax.set_title("Precision–recall curves from outer-fold predictions",fontweight="bold")
ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); save(fig,"figure7_precision_recall_curve.png")

captions=[
"Figure 1. Architecture of the proposed explainable labour-market intelligence framework. The ML component estimates whether an occupation–skill pair is observed in the available job-posting data.",
"Figure 2. Five most frequently observed skills for each target occupation, measured by relevant job-posting counts in the processed NYC dataset.",
"Figure 3. Distribution of skill-gap priority categories produced by the selected prevalence-based prioritisation rule.",
"Figure 4. Number of distinct prioritised skill-gap records for each target occupation.",
"Figure 5. Precision, recall and F1-score for the baseline at threshold 0.50, baseline with nested threshold selection, and class-weighted model with nested threshold selection.",
"Figure 6. Confusion matrices from pooled outer-fold predictions. Nested-threshold panels use thresholds selected within the corresponding outer fold's inner training procedure.",
"Figure 7. Precision–recall curves computed from pooled outer-fold probabilities for baseline and class-weighted Logistic Regression."
]
(OUT/"figure_captions.txt").write_text("\n\n".join(captions)+"\n",encoding="utf-8")
print("\nCaptions:",OUT/"figure_captions.txt")
print("\nAll figures saved to:",OUT)
