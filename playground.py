import pandas as pandas
import cliffsDelta.cliffsDelta as cd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
import scipy.stats as stats
import powerlaw
import pystan
import statsmodels.api as sm
import find_fit
from data_clean import LoadData as ld


# Query strings from the authors's replication package.


# what does data look like
#sns.stripplot(df['role'],df['cbo'], data=df, jitter=True)
ax = sns.violinplot(ld.cbo_pop['role'],np.log(ld.cbo_pop['cbo']+0.1), data=ld.spring,)
ax.set_ylabel('log(cbo)')
plt.show()

mann_dict = {}
cliff_dict = {}
for role in ld.roles[1:]:
    for role2 in ld.roles[1:]:
        if role is role2:
            pass
        first = ld.cbo_pop[(ld.cbo_pop.role == role)]
        other = ld.cbo_pop[(ld.cbo_pop.role == role2)]
        key = role +'-'+ role2
        mann_dict[key] = stats.mannwhitneyu(first['cbo'],other['cbo'])
        cliff_dict[key] = cd.cliffsDelta(first['cbo'],other['cbo'])

delta, effsize = cd.cliffsDelta(ld.cbo_controller['cbo'],ld.cbo_pop_controller['cbo'])
statistic, pvalue = stats.mannwhitneyu(ld.cbo_pop_controller['cbo'],ld.cbo_controller['cbo'])

# fit a theoretical distribution
# model histogram with Seaborn fit
ax = sns.distplot(ld.data, kde=False, fit=scipy.stats.norm)
ax = sns.distplot(ld.data, kde=False, fit=scipy.stats.lognorm)
# shape, loc, scale = stats.lognorm.fit(data) # some lognormal Scipy weirdness
# res2 = stats.probplot(data, dist=stats.lognorm, sparams=(shape, loc, scale), plot=plt)
# res = stats.probplot(data, dist=stats.norm, plot=plt)
# res = stats.probplot(np.log(data+0.1), dist=stats.norm, plot=plt)
plt.show()

# stats.lognorm.sf(data,sparams=(shape, loc, scale) ) # some lognormal Scipy weirdness

# # powerlaw package. Herraiz et al described CBO as a 'double powerlaw', namely a log-normal followed by pareto
# # note that CBO is discrete (only integer values possible)
# fit = powerlaw.Fit(data, discrete=True)
# fig = powerlaw.plot_pdf(data, color = 'b')
# fit.plot_ccdf(color = 'r', linewidth = 2, ax = fig) # complementary CDF
# # calculate the portion of data to fit.
# # see also https://github.com/Astroua/plndist/blob/master/DPLogNorm.ipynb for the Double Pareto LogNormal distribution
# plt.show()

# using StatModels
fig1 = sm.qqplot(ld.data, stats.norm, fit=True, line='45')
fig1.suptitle('Normal QQ')
fig2=sm.qqplot(ld.data, stats.lognorm, fit=True, line='45')
fig2.suptitle('LogNormal QQ')
fig3 = sm.qqplot(ld.data, stats.powerlaw, fit=True, line='45')
fig3.suptitle('Powerlaw QQ')
plt.show()

# using SSE from Stackoverflow
find_fit.plot_fits(ld.data, title='CBO scores with best fit distribution', xlab='CBO score')

