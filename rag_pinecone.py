# %%
import pandas as pd
import sys
sys.path.append('/teamspace/studios/this_studio/ConferenceGeneTargets')
from RAG_term_normalisation.vectorstore_gene_synonyms import VectorStore_genes
from dotenv import load_dotenv
#%%
load_dotenv()

tmp = VectorStore_genes()
tmp.create_or_load_index()



# %%

abstract = "#0002T cells selected from lymph node acquisition for adoptive cell therapyin NSCLC.\nTatiana Delgado Cruz1, Yanping Yang2, Rachel Honigsberg3, Geoffrey Markowitz4, Nasser K Altorki5, Vivek Mittal4, Moonsoo M. Jin6, Jonathan Villena-\nVargas4\n1Weill Cornell Medicine/Sandra and Edward Meyer Cancer Center, New York, NY,2Radiology, Houston Methodist, Houston, TX,3Meinig School of Biomedical\nEngineering, Weill Cornell, ithaca, NY,4Cardiothoracic Surgery, Weill Cornell Medicine/Sandra and Edward Meyer Cancer Center, New York,\nNY,5Cardiothoracic surgery, Weill Cornell Medicine/Sandra and Edward Meyer Cancer Center, New York, NY,6Radiology, Houston Methodist Research\nInstitute, New York, NY\n\nBackground: A primary limitation of PD-1 inhibitors in non-small cell lung cancer (NSCLC) arises from their inability to act on 'cold' tumors without tumor-reactive\nT cells, necessitating alternative approaches.  Adoptive cell therapy (ACT) using (CAR)-engineered cells, strives to enhance antitumor immunity but faces\nseveral challenges such as identifying safe antigens, tumor heterogeneity, antigen escape, cell trafficking, and T cell persistence. To address these issues, we\nexplored a new source of T cells from the benign tumor draining lymph nodes (tdLNs) of NSCLC patients. Our initial results have shown that tdLNs is a reservoir\nfor of tumor-relevant 'stem-like' T cells. We posit that employing these pluripotent T cells for ACT could achieve significant tumor rejection in NSCLC.\nMethods: Resected tumors, tdLN, non-draining (ndLN), and blood from NSCLC patients, as well as a syngeneic murine lung cancer model (344SQ), underwent\nanalysis. T cells were profiled using flow cytometry, cytotoxicity and proliferation assays. TCR and single cell (sc)RNA sequencing assessed clonal expansion,\ndiversity, and transcriptional profiles of tumor-relevant T cells. T cells were then transduced with an ICAM-1 targeting CAR, in vivo efficacy was evaluated in an\nA549 murine lung cancer model.\nResults: T cell subsets with stem-cell memory characteristics, as indicated by PD-1+, TCF1hi, CXCR5+, and CD8+ expression, which were not significantly\nfound in the tumor or PB. These T cells exhibited progenitor-like transcriptional signatures, enhanced SELL and TCF-1 expression, fewer exhaustion markers,\nand superior in vitro proliferation compared to TILs from both a murine lung cancer model and patient-derived tissues. scRNA sequencing, coupled with TCR\n\"tumor matching\" (TM) techniques, exposed a rich clonal diversity of tumor-relevant clones within tdLNs, which showcased a broader transcriptional memory\nprofile and distinct CD4+ and CD8+ phenotypes. Upon analyzing the top 100 expanded (n>3) TM clones, 47 featured the presence of tdLN-derived T cells,\ncovering progenitor, stem cell-like, and central memory clusters. T cell subsets were then transduced with a CAR targeting ICAM-1\u2014a cell surface protein\nfrequently overexpressed in NSCLC tumors. Manufacturing protocol yielded high transduction efficiency and T cell expansion within two weeks in 6/6 patients,\nconsistent with PB-derived CAR T cells and on par with the optimal dosing requirements of an ICAM-1 CAR Phase I trial (NCT04420754). tdLN-CAR T cells\nshowed potent antitumor efficacy compared to the control in an aggressive NSCLC murine model (Median survival 103d vs 66d; respectively; p=0.006).\nConclusions: This represents the first reported use of T cells from tdLN for genetically engineered ACT. The data indicate that modifying antigen-experienced,\nstem-like T cells from tdLN with CAR is a promising and efficient method in a NSCLC murine model, warranting further research."



tmp.retrieve('gene-index',abstract)

# %%

# fname_genes = 'data/RAG_LLM/features/genes_synonyms.pq'

# df_synonyms = pd.read_parquet(fname_genes)

# # %%

# df_descriptions = df_synonyms[['Symbol', 'description']]
# # rename description to synonym
# df_descriptions = df_descriptions.rename(columns={'description': 'Synonyms'})

# df_synonyms = df_synonyms[['Symbol', 'Synonyms']]
# #drop '-' and ''
# df_synonyms = df_synonyms[(df_synonyms['Synonyms'] != '-') & (df_synonyms['Synonyms'] != '')]
# df_descriptions = df_descriptions[(df_descriptions['Synonyms'] != '-') & (df_descriptions['Synonyms'] != '')]
# combined_df = pd.concat([df_synonyms, df_descriptions], axis=0).drop_duplicates().reset_index(drop=True)

# # %%
# #prepare_gene_synonyms

# # calculate embeddings for Synonyms


# # get one row per synonym/ description


# # data = [
# #     {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
# #     {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
# #     {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
# #     {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
# #     {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
# #     {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
# # ]

# vectorstore = VectorStore_pc()

# vectorstore.create_or_load_index("test-index", data)