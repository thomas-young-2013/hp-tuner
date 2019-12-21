from mfes.utils.properties import Properties

prop = Properties('./conf/conf.properties')
P_LOC = prop.get_property('P_LOC')

MAXINT = 2 ** 31 - 1
MINIMAL_COST_FOR_LOG = 0.00001 
MAX_CUTOFF = 65535
