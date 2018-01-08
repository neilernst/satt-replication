import pystan
import matplotlib.pyplot as plt
import pandas
import numpy as np

roles = ["none","controller", "dao", "service", "entity", "component"]
roles_num = range(len(roles))
role_dict = dict(zip(roles_num, roles))
spring_columns = ["project", "file", "class", "type", "test", "role", "cbo", "dit", "lcom", "noc", "nom", "rfc", "wmc"]
# load data
# N = 63525
spring = pandas.read_csv('data/metrics/spring.csv', header=None, names=spring_columns)
# N = 60701
sloc = pandas.read_csv('data/metrics/spring-sloc2-java.csv', usecols=['loc','file'])

# inner = drop NaN values
# N = 59760
final = spring.merge(sloc, on='file', how='inner')
# final.replace({'role': role_dict}, inplace=True)
# Strict replication would need to use DataFrame 'spring' and not 'final', due to merge
cbo_pop = final.loc[:,['test','type','cbo','role','loc']]
cbo_pop_controller = cbo_pop.query("test == 0 and type == 'class' and role != 'controller'")

cbo_controller = cbo_pop.query("role == 'controller'")
data = pandas.Series(cbo_controller['cbo'])
final.head()
projects = final.project.str.strip().unique()
proj_num = len(projects)
project_lookup = dict(zip(projects, range(proj_num)))
project = final['proj_code'] = final.project.replace(project_lookup).values
cbo = final.cbo
final['log_cbo'] = log_cbo = np.log(cbo + 0.1).values
sloc = final['loc']
final['log_sloc'] = log_sloc = np.log(sloc + 0.1).values
role_code = final.role

# reshape the Role field into onehot encoding (binarized)
# an array of size 6 where a 1 indicates True for each role
import sklearn.preprocessing as pp
lb = pp.LabelBinarizer().fit(role_code.values.reshape(-1,1))
binarized = lb.transform(role_code)
temp = pandas.DataFrame(binarized,columns=roles)
#using indices assumes equal size
final = final.merge(temp,left_index=True,right_index=True)

pooled_data = """
data {
  int<lower=0> N; 
  vector[N] x;
  vector[N] y;
}
"""
pooled_parameters = """
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
"""
pooled_model = """
model {
  y ~ normal(beta[1] + beta[2] * x, sigma);
}
"""
pooled_data_dict = {'N': len(log_cbo),
                    'x': final.controller,
                    'y': log_cbo}


compiled_model = pystan.stanc(model_code=pooled_data + pooled_parameters + pooled_model)
model = pystan.StanModel(stanc_ret=compiled_model)
# pooled_fit = model.sampling(data=pooled_data_dict, iter=1000, chains=2)
#
# pooled_sample = pooled_fit.extract(permuted=True)
# b0, m0 = pooled_sample['beta'].T.mean(1)
# # plt.scatter(final.controller, log_cbo)
# import seaborn as sns
# ax = sns.violinplot(final.controller, log_cbo)
# xvals = np.linspace(-0.2, 1.2)
# # here "0" means "not controller" which is a bit different from the original paper
# ax.plot(xvals, m0*xvals+b0, 'r--')

# unpooled
unpooled_model = """data {
  int<lower=0> N; 
  int<lower=1,upper=120> project[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[120] a;
  real beta;
  real<lower=0,upper=100> sigma;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = beta * x[i] + a[project[i]];
}
model {
  y ~ normal(y_hat, sigma);
}"""

unpooled_data = {'N': len(log_cbo),
               'project': project+1, # Stan counts starting at 1
                 'x': final.controller,
                 'y': log_cbo}

compiled_pooled_model = pystan.stanc(model_code= unpooled_model)
unpooled_model = pystan.StanModel(stanc_ret=compiled_pooled_model)
unpooled_fit = unpooled_model.sampling(data=unpooled_data, iter=1000, chains=2)

unpooled_sample = unpooled_fit.extract(permuted=True)
b0, m0 = unpooled_sample['beta'].T.mean(1)