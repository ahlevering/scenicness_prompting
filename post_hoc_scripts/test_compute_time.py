import timeit
import torch

examples_early = torch.zeros([10000, 500000], dtype=torch.float16)
times = timeit.repeat(lambda: torch.softmax(torch.zeros([10000, 500000], dtype=torch.float16), dim=0).T * torch.ones([10000]), repeat=10, number=1)
del examples_early
print(f"Average early ensembling execution time: {torch.mean(times)} seconds")

examples_late = torch.zeros([4, 500000], dtype=torch.float16)
def time_late():
    late_rating_sums = []
    for i in range(0,2500):
        late = torch.softmax(torch.zeros([4, 500000], dtype=torch.float16), dim=0).T * torch.ones([4])
        late_rating_sums.append(torch.sum(late, dim=1))
    late_sum = torch.sum(torch.stack(late_rating_sums))

times = timeit.repeat(time_late, repeat=10, number=1)
print(f"Average late ensembling execution time: {torch.mean(times)} seconds")