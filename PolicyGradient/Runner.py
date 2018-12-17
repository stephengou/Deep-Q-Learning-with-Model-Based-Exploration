import subprocess
import sys


PG_Algo = 'PG_MonteCarlo'
# PG_Algo = 'PG_A2C'

for scriptInstance in range(1,6):
    sys.stdout = open('{}_Result{}.txt'.format(PG_Algo, scriptInstance), 'w')
    subprocess.check_call(['python', PG_Algo+".py"], \
                          stdout=sys.stdout, stderr=subprocess.STDOUT)
