# import biom
# import skbio
import numpy as np
import pandas as pd
from deicode.matrix_completion import MatrixCompletion
from deicode.preprocessing import rclr

from scipy.linalg import svd

DEFAULT_RANK = 3
# DEFAULT_MSC = 500
# DEFAULT_MFC = 10
# DEFAULT_MFF = 0
DEFAULT_ITERATIONS = 5

def rpca(table,
         n_components = DEFAULT_RANK,
         # min_sample_count = DEFAULT_MSC,
         # min_feature_count = DEFAULT_MFC,
         # min_feature_frequency = DEFAULT_MFF,
         max_iterations = DEFAULT_ITERATIONS):
    """Runs RPCA with an rclr preprocessing step.

    """
    # get shape of table
    n_samples,n_features = table.shape

    # # filter sample to min seq. depth
    # def sample_filter(val, id_, md):
    #     return sum(val) > min_sample_count

    # # filter features to min total counts
    # def observation_filter(val, id_, md):
    #     return sum(val) > min_feature_count

    # # filter features by N samples presence
    # def frequency_filter(val, id_, md):
    #     return (np.sum(val > 0) / n_samples) > (min_feature_frequency / 100)

    # filter and import table for each filter above
    # table = table.filter(observation_filter, axis='observation')
    # table = table.filter(frequency_filter, axis='observation')
    # table = table.filter(sample_filter, axis='sample')
    # table = table.to_dataframe().T
    # check the table after filtering
    # if len(table.index) != len(set(table.index)):
    #     raise ValueError('Data-table contains duplicate indices')
    # if len(table.columns) != len(set(table.columns)):
    #     raise ValueError('Data-table contains duplicate columns')
    # Robust-clt (rclr) preprocessing and OptSpace (RPCA)
    opt = MatrixCompletion(n_components=n_components,
                           max_iterations=max_iterations).fit(rclr(table))
    # get new n-comp when applicable
    n_components = opt.s.shape[0]
    # get PC column labels for the skbio OrdinationResults
    rename_cols = ['PC' + str(i + 1) for i in range(n_components)]
    # get completed matrix for centering
    X = opt.sample_weights @ opt.s @ opt.feature_weights.T
    # center again around zero after completion
    X = X - X.mean(axis=0)
    X = X - X.mean(axis=1).reshape(-1, 1)
    # re-factor the data
    u, s, v = svd(X)
    # only take n-components
    u = u[:, :n_components]
    v = v.T[:, :n_components]
    # calc. the new variance using projection
    p = s**2 / np.sum(s**2)
    p = p[:n_components]
    s = s[:n_components]
    # save the loadings
    feature_loading = pd.DataFrame(v, index=table.columns,
                                   columns=rename_cols)
    sample_loading = pd.DataFrame(u, index=table.index,
                                  columns=rename_cols)
    # % var explained
    proportion_explained = pd.Series(p, index=rename_cols)
    # get eigenvalues
    eigvals = pd.Series(s, index=rename_cols)

    # if the n_components is two add PC3 of zeros
    # this is referenced as in issue in
    # <https://github.com/biocore/emperor/commit
    # /a93f029548c421cb0ba365b4294f7a5a6b0209ce>
    # discussed in DEICODE -- PR#29
    # if n_components == 2:
    #     feature_loading['PC3'] = [0] * len(feature_loading.index)
    #     sample_loading['PC3'] = [0] * len(sample_loading.index)
    #     eigvals.loc['PC3'] = 0
    #     proportion_explained.loc['PC3'] = 0

    # save ordination results
    # short_method_name = 'rpca_biplot'
    # long_method_name = '(Robust Aitchison) RPCA Biplot'
    # ord_res = skbio.OrdinationResults(
    #     short_method_name,
    #     long_method_name,
    #     eigvals.copy(),
    #     samples=sample_loading.copy(),
    #     features=feature_loading.copy(),
    #     proportion_explained=proportion_explained.copy())
    # save distance matrix
    dist_res = p= pd.DataFrame(opt.distance, index=sample_loading.index,
                                   columns=sample_loading.index)#skbio.stats.distance.DistanceMatrix(
        # opt.distance, ids=sample_loading.index)

    return dict(eig=eigvals, smp=sample_loading, feat=feature_loading, pexp=proportion_explained), dist_res#ord_res, dist_res


def auto_rpca(table,
              # min_sample_count: int = DEFAULT_MSC,
              # min_feature_count: int = DEFAULT_MFC,
              # min_feature_frequency: float = DEFAULT_MFF,
              max_iterations = DEFAULT_ITERATIONS):
    """Runs RPCA but with auto estimation of the
       rank peramater.
    """
    ord_res, dist_res = rpca(table,
                             n_components='auto',
                             # min_sample_count=min_sample_count,
                             # min_feature_count=min_feature_count,
                             # min_feature_frequency=min_feature_frequency,
                             max_iterations=max_iterations)
    return ord_res, dist_res
