import pandas as pandas


class LoadData:
    # Join SLOC and other metrics on filename
    # from SATT replication data. I assume this order is == to index
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
    final.replace({'role': role_dict}, inplace=True)
    # Strict replication needs to use DataFrame 'spring' and not 'final', due to merge
    cbo_pop = final.loc[:,['test','type','cbo','role','loc']]
    cbo_pop_controller = cbo_pop.query("test == 0 and type == 'class' and role != 'controller'")

    cbo_controller = cbo_pop.query("role == 'controller'")
    data = pandas.Series(cbo_controller['cbo'])

    # role counts of merged data
    # print(df.groupby('role')['file'].nunique())
    # component      2048
    # controller     3024
    # dao            1265
    # entity         1650
    # none          49001
    # service        2773
    # plt.interactive(False)
    # print(df.isnull().sum())

