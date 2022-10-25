from os import system

for num_cores in [4,8,12,16,24,32]:
    batchScript = 'isca_job_cores_%s.batch'%(num_cores)
    f1 = open('empty_isca_job','r')
    f2 = open(batchScript,'w')
    f2.write(f1.read()%(num_cores,
                        num_cores,
                        num_cores,
                        num_cores,
                        num_cores))
    f1.close(); f2.close()
    system('sbatch %s'%(batchScript))


