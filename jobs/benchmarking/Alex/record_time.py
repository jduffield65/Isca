import sys as sys
import os
from datetime import datetime
reformatted_GMT_timestamp = datetime.utcnow().strftime('[%Y,%m,%d,%H,%M,%S]')
outputName = sys.argv[1]
outputLine = sys.argv[2]
f = open(outputName, 'a')
f.write('%10s = %s #[Y,m,d,H,M,S]\n'%(outputLine+' '*10, reformatted_GMT_timestamp))
f.write(os.environ['NMONTHS'])
f.close()

