# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from dowhy import CausalModel


# %%
data = pd.read_csv("./data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")


# %%
male_df = data[data['segment'] != 'Womens E-Mail']
male_df.shape


# %%
male_df['treatment'] = male_df['segment'].map(lambda x: 1 if x == 'Mens E-Mail' else 0)
male_df.head(3)


# %%
# 集計
summary_by_segment = pd.pivot_table(
    data=male_df,
    values=['conversion', 'spend', 'visit'],
    index=['treatment'],
    aggfunc={'conversion': np.mean, 'spend': np.mean, 'visit': np.ma.count}
)

summary_by_segment.columns = ['conversion_rate', 'spend_mean', 'count']
summary_by_segment


# %%
# バイアスのあるデータの準備
treatment_data = male_df[male_df['treatment'] == 1]
control_data = male_df[male_df['treatment'] == 0]

treatment_biased = treatment_data.drop(treatment_data[~(
    (treatment_data['history'] > 300) |
    (treatment_data['recency'] < 6) |
    (treatment_data['recency'] == 'Multichannel')
)].sample(frac=0.5, random_state=1).index)

control_biased = control_data.drop(control_data[
    (control_data['history'] > 300) |
    (control_data['recency'] < 6) |
    (control_data['recency'] == 'Multichannel')
].sample(frac=0.5, random_state=1).index)

biased_data = pd.concat([treatment_biased, control_biased])
biased_data.head(3)


# %%
# バイアスのあるデータの集計と有意差の検定
summary_by_segment_biased = pd.pivot_table(
    data=biased_data,
    values=['conversion', 'spend', 'visit'],
    index=['treatment'],
    aggfunc={'conversion': np.mean, 'spend': np.mean, 'visit': np.ma.count}
)
summary_by_segment_biased.columns = ['conversion_rate', 'spend_mean', 'count']


# %%
summary_by_segment_biased


# %%
import numpy as np
struct_data = biased_data.copy()

non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)


# %%

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])

struct_data["treatment"] = struct_data["treatment"].astype(bool)

struct_data.head(5)


# %%
G = nx.DiGraph()  # 有向グラフ (Directed Graph)

# 辺の追加 (頂点も必要に応じて追加されます)
G.add_edges_from([ ("channel", "spend"), ("channel", "treatment"), ("history", "spend"), ("history", "treatment"), ("recentry", "spend"), ("recentry", "treatment"), ("treatment", "spend")])

causal_diagram = "".join(nx.generate_gml(G))

nx.nx_agraph.view_pygraphviz(G, prog='fdp')  # pygraphvizが必要


# %%
model=CausalModel(
        data = struct_data,
        treatment="treatment",
        outcome="spend",
        graph=causal_diagram
        )


# %%
model.view_model()


# %%

from IPython.display import Image, display
display(Image(filename="causal_model.png"))


# %%

identified_estimand = model.identify_effect()
print(identified_estimand)


# %%

causal_estimate = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.propensity_score_weighting",
                                            target_units = "ate",
                                            method_params={"weighting_scheme":"ips_weight"})
print(causal_estimate)
print("Causal Estimate is " + str(causal_estimate.value))


# %%
res_random=model.refute_estimate(identified_estimand, causal_estimate, method_name="random_common_cause")
print(res_random)


# %%
res_unobserved=model.refute_estimate(identified_estimand, causal_estimate, method_name="add_unobserved_common_cause",
                                     confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear",
                                    effect_strength_on_treatment=0.01, effect_strength_on_outcome=0.02)
print(res_unobserved)


# %%

res_placebo=model.refute_estimate(identified_estimand, causal_estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)


# %%



