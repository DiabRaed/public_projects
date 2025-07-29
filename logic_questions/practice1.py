# %%
# This is a script to practice questions for IBM.

#word counter

from collections import Counter
import enum
import re 
def most_repeated_word(text: str):
    if not text:
        return ""
    
    words = re.findall(r'\b\w+\b', text.lower()) 
    
    counts=Counter(words )

    most_common=sorted(counts.items(),key=lambda x: (-x[1], x[0]))

    return most_common[0][0]
# %%
most_repeated_word("This is not what I expected, but what is?") 
# %%


import numpy as np 
import statistics

def remove_outliers(data: list[int]):
    if not all(isinstance(item, (int,float)) for item in data):
        raise ValueError("Input must contain numberes only!")
    # mean=np.mean(data)
    # std=np.std(data)
    mean=statistics.mean(data)
    std=statistics.stdev(data)
    good_numbers=[]
    for number in data:
        if mean-std <= number <= mean+std: 
            good_numbers.append(number)
        else:
            pass 
    return good_numbers 

remove_outliers([10,32,2,13,32,324,138])
# %%


#most common element in a list

def most_common_element(data: list[int]):
    """ This code returns the most common number (integer or float) 
    in a list. If a tie, return the smaller number """

    if len(data)==0:
        print("The list is empty. Please provide a good list")

    else:
        numbers=Counter(data)

        sorted_list=sorted(numbers.items(),key=lambda x:(-x[1],x[0]))


        return sorted_list[0][0]

most_common_element([1,1,2,3,2,3,2])
# %%

#string compression

def compress_string(s: str) -> str:
    """This code compresses strings by replacing consecutive duplicate characters
    with the count. e.g aaabbbcc becomes a3b3c2"""
    if not s:
        return ""

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + str(count))
            count = 1

    # Append the final group
    result.append(s[-1] + str(count))
    return ''.join(result)

print(compress_string("aaabbbcc"))  # a3b3c2
print(compress_string("aba"))       # a1b1a1
print(compress_string(""))          # ""

# %%

#Rotate a list
def rotate_list(lst: list[int], k: int) -> list[int]:
    """
    Rotates the list to the right by k steps.
    e.g., [1,2,3,4,5], k=2 → [4,5,1,2,3]
    """
    if not lst:
        raise ValueError("Please provide a proper list")
        
    elif k>len(lst) or k <0:
        raise ValueError("k must obey 0 < k < len(lst)")

    #take the index then add k to it. The k index becomes 0 index
    new_list=[]
    for idx,i in enumerate((lst)):
        # print(lst[idx-k])
        i=lst[idx-k]
        new_list.append(i)
    return new_list

# print(rotate_list([1,2,3,4],k=-1))

print(rotate_list([1, 2, 3, 4, 5], 2))  # [4, 5, 1, 2, 3]
print(rotate_list([1, 2, 3], 3))        # [1, 2, 3] (k=length)
print(rotate_list([1, 2, 3], 0))        # [1, 2, 3]

# or the one chatgpt provided

def rotate_listt(lst: list[int], k:int):
    if not lst:
        raise ValueError("Please provide a non-empty list")

    n=len(lst)
    k=k%n 
    new_list=lst[-k:] + lst[:-k]
    print(lst[-k:],lst[:-k])

rotate_listt([1,2,3,4,5,6],k=1)
# %%

#Standardize values
import statistics
import matplotlib.pyplot as plt
def standardize(data: list[float]):
    """This script standardizes a list of data, such that it has 
    a mean of 0 and a standard deviation of 1"""

    if not data:
        raise ValueError("Please provie a non-empty list")
    
    mean=statistics.mean(data)
    std=statistics.stdev(data)
    if std == 0:
        raise ValueError("Standard deviation is zero; data must not be constant")

    standardized_list=[((x-mean)/std) for x in data]

    return standardized_list

standardize([i for i in np.linspace(-7,7,10)])
# %%


#moving average 

def moving_average(data: list[float], window_size: int) -> list[float]:
    """
    Return the moving average of the list with a given window size.
    e.g., [1,2,3,4], window=2 → [1.5, 2.5, 3.5]
    """
    if not data:
        raise ValueError('Please provide a proper non-empty list')
    

    moving_list=[]
    for i in range(len(data)-window_size+1):
        window=data[i:i+window_size]
        # print(i)
        moving_list.append(sum(window)/window_size)
    
    return moving_list

moving_average([1,2,3,4],window_size=3)

# %%

#Categorical Counter 

def category_counts(data: list[str]):
    """Takes a list of string labels and returns a frequency dictionary
    """

    if not data:
        raise ValueError("Please provide a non-empty list")
    
    return  dict(Counter(data))


category_counts(['cat','dog','apple','cat','dog'])
# %%
