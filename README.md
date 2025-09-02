# PCA_gen_AI
For the base code check https://github.com/marziof/PLM-gen-DCA . This read me we only treat the modifications made on the base code namely the different trainer methods for different models and a new generation models.

*Trainers/models*
All the trainers can be found in attention.py. The main modifications to those functions are the addition of the PCA coordinates to the train sequences, and the choice of the model used.
trainer(): Uses the normal AttentionModel (no PCA components). A small modification was made: masking after the softmax instead of before to match the Julia code
trainer_PCA_comp_brute_force(): Add PCA components and treats them as extra amino acides in the chain expanding the J tensor to (q',q',L+N_PCA,L+N_PCA) with q'=max(n_bins,q). This trainer uses the same AttentionModel just with different dimensions than the prvious one
trainerCondJ(): Add PCA comp as well. Only uses one tensor J but with less dimensions than the last one. J has a shape of (q,q+nbins,L,L+N_PCA). The model used is ModelPCACond.
trainer_PCA_comp_2_model(): trains a second tensor G (not the J for the sequence self interaction), that discribes the interaction between the PCA components and the aa chain. This model needs the QKV tensors used to calculate the J tensor to calculate the loss properly.
trainer_PCA_comp_2_model_once(): calculates the same as the one before but it finds J and G at the same time so no need to give it the QKV tensors.
ar_trainer(): Implements an autoregressive approach to compare with the Julia.
The trainings are done in the file train_jdoms_new.ipynb
*Generation methods*
In plm_model.py, we modified the classes to generate with the added PCA components and vectorized the functions to generate faster. We also added a new class called BatchSequencePLMvec that creates N indepedent sequences to be evolved seperatly to guarantee that the final sequences (one from each chain) are independent. We make all the sequences evolve for n_steps (usually double the correlation length) to have final sequences.
In plm_gen_methods.py you can find functions that call the generation classes and generate the sequences and saves them.
In the generation methods, one can specify the beta_PCA, the sampling temperature related to the PCA component.
The generations are done in the file vectorized_gen_test.ipynb

*Results visualization*.
The main plots can  be found in plm_gen_PCA_plots3.ipynb. We use PCA and frequency/couple correlations to compare the generated and train sequences. For high PCA dimension, we use Kmeans method to detect the clusters in the PCA space (so we can have a target to generate at). Then we compare the train sequences in the cluster to the generated ones using root mean square distance in the PCA space and the Wasserstein distance for each component. We also plot the frequencies of the generated sequences against the whole data set and against the ones in the chosen cluster. 
The frequency methods this time were modified to use the weights of the train sequences as well to calculate the frequencies of the train sequences.





