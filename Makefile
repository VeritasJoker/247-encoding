# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")
DT := ${USR}

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := tfs
# {podcast | tfs}

############## tfs electrode ids ##############
# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)
BC := 

# 676 Electrode IDs
SID := 676
E_LIST := $(shell seq 1 125)
BC := --bad-convos 38 39

# 717 Electrode IDs
# SID := 7170
# E_LIST := $(shell seq 1 256)
# BC :=

# 798 Electrode IDs
# SID := 798
# E_LIST := $(shell seq 1 198)
# BC :=

# Sig file will override whatever electrodes you choose
SIG_FN := 
# SIG_FN := --sig-elec-file tfs-sig-file-625-sig-1.0-comp.csv tfs-sig-file-625-sig-1.0-prod.csv tfs-sig-file-676-sig-1.0-comp.csv tfs-sig-file-676-sig-1.0-prod.csv tfs-sig-file-7170-sig-1.0-comp.csv tfs-sig-file-7170-sig-1.0-prod.csv tfs-sig-file-798-sig-1.0-comp.csv tfs-sig-file-798-sig-1.0-comp.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-sig-1.0-comp.csv tfs-sig-file-625-sig-1.0-prod.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-top-0.08-comp.csv tfs-sig-file-625-top-0.08-prod.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-region-ifg.csv tfs-sig-file-625-region-ifg.csv tfs-sig-file-676-region-ifg.csv tfs-sig-file-676-region-ifg.csv tfs-sig-file-7170-region-ifg.csv tfs-sig-file-7170-region-ifg.csv tfs-sig-file-798-region-ifg.csv tfs-sig-file-798-region-ifg.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-all.csv tfs-sig-file-625-all.csv tfs-sig-file-676-all.csv tfs-sig-file-676-all.csv tfs-sig-file-7170-all.csv tfs-sig-file-7170-all.csv tfs-sig-file-798-all.csv tfs-sig-file-798-all.csv


# podcast electrode IDs
# SID := 777
# SID := 661
# E_LIST :=  $(shell seq 1 115)
# SID := 662
# E_LIST :=  $(shell seq 1 100)
# SID := 717
# E_LIST :=  $(shell seq 1 255)
# SID := 723
# E_LIST :=  $(shell seq 1 165)
# SID := 741
# E_LIST :=  $(shell seq 1 130)
# SID := 742
# E_LIST :=  $(shell seq 1 175)
# SID := 743
# E_LIST :=  $(shell seq 1 125)
# SID := 763
# E_LIST :=  $(shell seq 1 80)
# SID := 798
# E_LIST :=  $(shell seq 1 195)
#

### podcast significant electrode list (if provided, override electrode IDs)
# SIG_FN := --sig-elec-file podcast_160.csv

PKL_IDENTIFIER := full
# {full | trimmed | full_masked_l_r | full_masked_l | full_masked_r}

# number of permutations (goes with SH and PSH)
NPERM := 1000

# Choose the lags to run for.
LAGS := {400000..500000..100} # lag400500k-100
LAGS := {-150000..150000..100} # lag60k-1k
LAGS := {-500..500..5} # lag500-5
LAGS := -300000 -250000 -200000 200000 250000 300000 # lag300k-50k
LAGS := -150000 -120000 -90000 90000 120000 150000 # lag150k-30k
LAGS := {-2000..2000..25} # lag2k-25
LAGS := {-10000..10000..25} # lag10k-25
LAGS := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000 # lag60k-10k

# Conversation ID (Choose 0 to run for all conversations)
CONVERSATION_IDX := 0

# Choose which set of embeddings to use
# {glove50 | gpt2-xl | blenderbot-small}
EMB := blenderbot
EMB := blenderbot-small
EMB := bert-large-uncased
EMB := roberta-large
EMB := bert-large-cased
EMB := glove50
EMB := gpt2-xl
EMB := whisper-tiny.en
CNXT_LEN := 1

# Choose the window size to average for each point
WS := 200

# Choose which set of embeddings to align with (intersection of embeddings)
ALIGN_WITH := blenderbot-small
ALIGN_WITH :=  glove50 gpt2-xl blenderbot-small
ALIGN_WITH := gpt2-xl
ALIGN_WITH := glove50 gpt2-xl
ALIGN_WITH := $(EMB)

# Choose layer of embeddings to use
# {1 for glove, 48 for gpt2, 16 for bbot, 24 for bert-lg}
LAYER_IDX := 0 1 2 3 4

# Choose whether to PCA (0 or for no pca)
PCA_TO := 50

# Specify the minimum word frequency (0 for 247, 5 for podcast)
MWF := 0

# Specify the number of folds (usually 10)
FN := 10

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
# SH := --shuffle
# PSH := --phase-shuffle

# Choose whether to normalize the embeddings
NM := l2
# {l1 | l2 | max}


# Token type (choose which tokens to take)
# all: use all tokens
# first: use all first tokens for each word
# last: use all last tokens for each word
# ave: use ave embedding of all tokens for each word
# root: use only tokens that are root
TT := root
TT := first
TT := last
TT := mean
TT := all

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := python
CMD := echo
CMD := sbatch submit1.sh
# {echo | python | sbatch submit1.sh}



############## Datum Modifications ##############

# 1. {no-trim}
#	if 'no-trim' is a substring of DM, do not trim datum words that have any lag \
outside of the conversation range (currently not used)
#	if 'no-trim' is not a substring of DM, datum will be trimmed based on maximum lag

# 2. {all, correct, incorrect, pred}
#	for all emb_type:
#	{all: choose all words}

#	for emb_type other than glove:
#	{correct: choose words correctly predicted by the model}
#	{incorrect: choose words incorrectly predicted by the model}

#	for all emb_type, use predictions from another emb_type by concat 'emb_type' and 'pred_type':
#	{gpt2-xl-corret: choose words correctly predicted by gpt2}
#	{gpt2-xl-incorret: choose words incorrectly predicted by gpt2}
#	{blenderbot-small-correct: choose words correctly predicted by bbot decoder}
#	{blenderbot-small-incorrect: choose words incorrectly predicted by bbot decoder}
#	{gpt2-pred: choose all words, for words incorrectly predicted by gpt2, use embeddings of the words \
actually predicted by gpt2} (only used for podcast glove)

# 3. {shift-emb, concat-emb}
# {shift-emb: shifts embeddings (eg, from n-1 to n)}
# {shift-emb1: shifts embeddings (eg, from n-1 to n)}
# {shift-emb2: shifts embeddings 2 times (eg, from n-1 to n+1)}
# ... etc
# {shift-embn: shifts embeddings (eg, from n-1 to n-2)}
# {shift-embn1: shifts embeddings (eg, from n-1 to n-2)}
# {shift-embn2: shifts embeddings 2 times (eg, from n-1 to n-3)}
# ... etc

# {concat-emb: concat embeddings (eg, n-1 + n)}
# {concat-emb2: concat embeddings (eg, n-1 + n + n+1)}
# ... etc

# 3. {everything else is purely for the result folder name}

DM := lag2k-25-incorrect
DM := lag2k-25-all
DM := lag60k-10k-all

############## Model Modification ##############
# {best-lag: run encoding using the best lag (lag model with highest correlation)}
# {pc-flip-best-lag: train on comp and test on prod using the best lag model, vice versa}
# {leave empty for regular encoding}
MM := best-lag
MM := pc-flip-best-lag
MM := 

#TODO: move paths to makefile

# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-encoding:
	mkdir -p logs
	$(CMD) scripts/$(FILE).py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--datum-emb-fn $(DS) \
		--sid $(SID) \
		--conversation-id $(CONVERSATION_IDX) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--window-size $(WS) \
		--word-value $(WV) \
		--npermutations $(NPERM) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		--fold-num $(FN) \
		--pca-to $(PCA_TO) \
		--layer-idx $(LAYER_IDX) \
		--token-type $(TT) \
		--datum-mod $(DM) \
		--model-mod $(MM) \
		$(BC) \
		$(SIG_FN) \
		$(SH) \
		$(PSH) \
		--normalize $(NM)\
		--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB)-$(DM) \
		--output-prefix $(USR)-$(WS)ms-$(WV);\


run-encoding-layers:
	mkdir -p logs
		for layer in $(LAYER_IDX); do\
			$(CMD) scripts/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--datum-emb-fn $(DS) \
				--sid $(SID) \
				--conversation-id $(CONVERSATION_IDX) \
				--electrodes $(E_LIST) \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				--fold-num $(FN) \
				--pca-to $(PCA_TO) \
				--layer-idx $$layer \
				--token-type $(TT) \
				--datum-mod $(DM) \
				--model-mod $(MM) \
				$(BC) \
				$(SIG_FN) \
				$(SH) \
				$(PSH) \
				--normalize $(NM)\
				--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB)-T$(TT)-$(DM)-mwf$(MWF)-l$$layer \
				--output-prefix $(USR)-$(WS)ms-$(WV);\
		done; \

# Recommended naming convention for output_folder
#--output-prefix $(USR)-$(WS)ms-$(WV); \

# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) scripts/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--sid $(SID) \
				--electrodes $$elec \
				--conversation-id $(CONVERSATION_IDX) \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				--pca-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--normalize $(NM) \
				--output-parent-dir $(PRJCT_ID)-$(PKL_IDENTIFIER)-$(EMB)-pca$(PCA_TO); \
				# --output-prefix ''; \
				# --job-id $(EMB)-$$jobid; \
		# done; \
	done;


# run-sig-encoding-slurm:
# 	mkdir -p logs
# 	for elec in $(E_LIST); do \
# 		# for jobid in $(shell seq 1 1); do \
# 			$(CMD) scripts/$(FILE).py \
# 				--project-id $(PRJCT_ID) \
# 				--pkl-identifier $(PKL_IDENTIFIER) \
# 				--sig-elec-file bobbi.csv \
# 				--emb-type $(EMB) \
# 				--context-length $(CNXT_LEN) \
# 				--align-with $(ALIGN_WITH) \
# 				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
# 				--window-size $(WS) \
# 				--word-value $(WV) \
# 				--npermutations $(NPERM) \
# 				--lags $(LAGS) \
# 				--min-word-freq $(MWF) \
# 				--pca-to $(PCA_TO) \
# 				$(SH) \
# 				$(PSH) \
# 				--output-parent-dir podcast-gpt2-xl-transcription \
# 				--output-prefix ''; \
# 				# --job-id $(EMB)-$$jobid; \
# 		# done; \
# 	done;


# pca-on-embedding:
# 	python scripts/tfsenc_pca.py \
# 			--sid $(SID) \
# 			--emb-type $(EMB) \
# 			--context-length $(CNXT_LEN) \
# 			--pca-to $(EMB_RED_DIM);


# Run erp for the given electrodes in one swoop

run-erp:
	mkdir -p logs
	$(CMD) scripts/tfserp_main.py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--datum-emb-fn $(DS) \
		--sid $(SID) \
		--conversation-id $(CONVERSATION_IDX) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--window-size $(WS) \
		--word-value $(WV) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		--layer-idx $(LAYER_IDX) \
		--datum-mod $(DM) \
		--normalize $(NM)\
		$(SIG_FN) \
		--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-erp-$(DM) \
		--output-prefix $(USR)-$(WS)ms-$(WV);\


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

########################## Regular Plotting Parameters ##########################
# LAGS_PLT: lags to plot (should have the same lags as the data files from formats)
# LAGS_SHOW: lags to show in plot (lags that we want to plot, could be all or part of LAGS_PLT)

# X_VALS_SHOW: x-values for those lags we want to plot (same length as LAGS_SHOW) \
(for regular encoding, X_VALS_SHOW should be the same as LAGS_SHOW) \
(for concatenated lags, such as type Quardra and type Final plots, X_VALS_SHOW is different from LAGS_SHOW)

# LAG_TKS: lag ticks (tick marks to show on the x-axis) (optional)
# LAT_TK_LABLS: lag tick labels (tick mark lables to show on the x-axis) (optional)

# Plotting for vanilla encoding (no concatenated lags)
LAGS_PLT := $(LAGS)
LAGS_SHOW := $(LAGS)
X_VALS_SHOW := $(LAGS_SHOW)
LAG_TKS := 
LAG_TK_LABLS :=

# Plotting for type Quardra (four different concatenated lags for 247)
# LAGS_PLT := {-300000..-150000..50000} -120000 -90000 {-60000..-20000..10000} {-10000..10000..25} {20000..60000..10000} 90000 120000 {150000..300000..50000}
# LAGS_SHOW := $(LAGS_PLT)
# X_VALS_SHOW := {-28000..-16000..2000} {-15000..-12000..1000} {-10000..10000..25} {12000..15000..1000} {16000..28000..2000}
# LAG_TKS := --lag-ticks {-28..28..2}
# LAG_TK_LABLS := --lag-tick-labels -300 -250 -200 -150 -120 -90 -60 -40 -20 {-10..10..2} 20 40 60 90 120 150 200 250 300

# Plotting for type Final (final plots for 247) 
# LAGS_PLT := {-60000..-20000..10000} {-10000..10000..25} {20000..60000..10000}
# LAGS_SHOW := $(LAGS_PLT)
# X_VALS_SHOW := -16000 -15000 -14000 -13000 -12000 {-10000..10000..25} 12000 13000 14000 15000 16000
# LAG_TKS := --lag-ticks {-16..16..2}
# LAG_TK_LABLS := --lag-tick-labels -60 -40 -20 {-10..10..2} 20 40 60

# zoomed-in version (from -2s to 2s)
# LAGS_SHOW := {-2000..2000..25}
# X_VALS_SHOW := {-2000..2000..25}
# LAG_TKS := 
# LAG_TK_LABLS :=

########################## Other Plotting Parameters ##########################
# Line color by (Choose what lines colors are decided by) (required)
# { --lc-by labels | --lc-by keys }

# Line style by (Choose what line styles are decided by) (required)
# { --ls-by labels | --ls-by keys }

# Split Direction, if any (Choose how plots are split) (optional)
# {  | --split horizontal | --split vertical }

# Split by, if any (Choose how lines are split into plots) (Only effective when Split is not empty) (optional)
# {  | --split-by labels | --split-by keys }

PLT_PARAMS := --lc-by labels --ls-by keys # plot for just one key (podcast plots)
PLT_PARAMS := --lc-by labels --ls-by keys --split horizontal --split-by keys # plot for prod+comp (247 plots)

# Figure Size (width height)
FIG_SZ:= 15 6
FIG_SZ:= 18 6

# Note: if lc_by = labels, order formats by: glove (blue), gpt2 (orange), bbot decoder (green), fourth label (red)

# Note: when providing sig elec files, provide them in the (sid keys) combination order \
For instance, if sid = 625 676, keys = prod comp \
sig elec files should be in this order: (625 prod)(625 comp)(676 prod)(676 comp) \
The number of sig elec files should also equal # of sid * # of keys


plot-new:
	rm -f results/figures/*
	python scripts/tfsplt_new.py \
		--sid 625 \
		--formats \
			'results/tfs/kw-tfs-full-625-whisper-tiny.en-Tall-lag10k-25-all-mwf0-l2/*/*_%s.csv' \
			'results/tfs/kw-tfs-full-625-whisper-tiny.en-Tall-lag10k-25-all-mwf0-l3/*/*_%s.csv' \
			'results/tfs/stock-1024-48/kw-tfs-full-625-gpt2-xl-lag10k-25-all/*/*_%s.csv' \
			'results/tfs/stock-1024-48/kw-tfs-full-625-gpt2-xl-lag10k-25-all-shift-emb/*/*_%s.csv' \
		--labels whisper-l2 whisper-l3 gpt2-n-1 gpt2-n \
		--keys comp prod \
		$(SIG_FN) \
		--fig-size $(FIG_SZ) \
		--lags-plot $(LAGS_PLT) \
		--lags-show $(LAGS_SHOW) \
		--x-vals-show $(X_VALS_SHOW) \
		$(LAG_TKS) \
		$(LAG_TK_LABLS) \
		$(PLT_PARAMS) \
		--outfile results/figures/tfs-625-whisper-gpt2-glove0.08.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


# HAS_CTX := --has-ctx
SIG_ELECS := --sig-elecs

CONDS := all correct incorrect
CONDS := all flip

plot_layers:
	rm -f results/figures/*
	python scripts/tfsplt_layer.py \
		--sid 625 \
		--layer-num 16 \
		--top-dir results/tfs/bbot-layers-625 \
		--modes comp prod \
		--conditions $(CONDS) \
		$(HAS_CTX) \
		$(SIG_ELECS) \
		--outfile results/figures/625-ericplots-bbot.pdf


# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------

SP := 1
SIG_LAGS := {-2000..2000..25}
SIG_LAGS := {-500..-100..25}

sig-test:
	rm -f results/figures/*
	python scripts/tfsmis_sigtest.py \
		--sid 798 \
		--formats \
			'results/tfs/20221201-gpt2-preds-top5/kw-tfs-full-798-glove50-lag10k-25-gpt2-xl-correct/*/*_%s.csv' \
			'results/tfs/20221201-gpt2-preds-top5/kw-tfs-full-798-glove50-lag10k-25-gpt2-xl-incorrect/*/*_%s.csv' \
		--labels correct incorrect \
		--emb glove \
		--keys prod comp \
		--values $(LAGS) \
		--sig-lags $(SIG_LAGS) \
		$(SIG_FN) \
		--sig-percents $(SP)


# make sure the lags and the formats are in the same order
LAGS1 := {-10000..10000..25}
LAGS2 := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000
LAGS3 := -150000 -120000 -90000 90000 120000 150000
LAGS4 := -300000 -250000 -200000 200000 250000 300000
# LAGS_FINAL := -300000 -60000 -30000 {-10000..10000..25} 30000 60000 300000 # final
LAGS_FINAL := -99999999 # select all the lags that are concatenated (quardra)


concat-lags:
	python scripts/tfsenc_concat.py \
		--format \
			'results/tfs/kw-tfs-full-798-glove50-lag10k-25-rand-aligned/kw-200ms-all-798/' \
			'results/tfs/kw-tfs-full-798-glove50-lag60k-10k-rand-aligned/kw-200ms-all-798/' \
		--lags \
			$(LAGS1) \
			$(LAGS2) \
		--lags-final $(LAGS_FINAL) \
		--output-dir results/tfs/plot-798-rand-double/kw-200ms-all-798/


# plot-autocor:
# 	$(CMD) scripts/test.py

