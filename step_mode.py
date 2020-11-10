def stepDiffusion(d_dataSet,diffusion_t,diffusion_d,mode='weight'):
    """
    :param
        d_dataSet:  divided dataset(also normalized).

        diffusion_t:    times of dissusion          ,like [1,3,5,2,1]
        diffusion_d:    each attributes' diffusion distance,like [[-0.2,-0.1,0,0.1,0.2], ...]
        # len(diffusion_d[i])=len(diffusion_t)

    :mode param:
        mode =='weight': transfer parameter of the weight into the algorithm for calculate.
        mode == 'brute':    repeat the record in dataSet according to the weight<int>.
    """
    ret_dataSet=[]
    if mode=='brute':
        for record in d_dataSet:
            for dd in range(len(diffusion_t)):
                record_t=[x+y[dd] for x in record for y in diffusion_d ]
                for t in range(diffusion_t[dd]):
                    ret_dataSet.append(record_t)
        print('brute diffused')
        return ret_dataSet

    if mode=='weight':
        for record in d_dataSet:
            for dd in range(len(diffusion_t)):
                record_t = [x + y[dd] for x in record for y in diffusion_d]
                ret_dataSet.append([record_t,diffusion_t[dd]])
        print('weight diffused')
        return ret_dataSet

if __name__=='__main__':
    from loadData import *
    dataSet = d_apyori_cookDataSet()
    dataSet.quickStart(fileName='test9_11.csv', haveHeader=True)
    ret_dataSet_brute=stepDiffusion(dataSet.d_data,[1,3,1],[[-0.1,0,0.1],]*len(dataSet.n_data[0]),mode='brute')
    #for i in ret_dataSet_brute:
    #    print(i)

    ret_dataSet_weight=stepDiffusion(dataSet.d_data,[1,3,1],[[-0.1,0,0.1],]*len(dataSet.n_data[0]),mode='weight')
    for i in ret_dataSet_weight:
        print(i)