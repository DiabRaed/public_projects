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
    print(sorted(Counter(data)))
    print(Counter(data).most_common(1))
    print(''.join(Counter(data).elements()))

    return  dict(Counter(data))


category_counts(['cat','dog','apple','cat','dog'])
# %%

#Let's compute accuray 

def compute_accuray(y_true:list[int], y_pred:list[int]):
    """This code computes the accuray in the following method
    accuracy=correct_predictions/total"""
    
    if len(y_true) != len(y_pred) or len(y_true) ==0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred should have the same non-zero length")

    # if not all(isinstance(item, (int,float)) for item in data):
    #     raise ValueError("Input must contain numberes only!")
    
    if not all(isinstance(item,(int,float)) for item in y_true ):
        raise ValueError("Input must contain numberes only!")

    if not all(isinstance(item,(int,float)) for item in y_pred ):
        raise ValueError("Input must contain numberes only!")
    #first, get the length
    n=len(y_true)

    #compare element by element. If true, add to count
    count=0

    for y_true_val,y_pred_val in zip(y_true,y_pred):
        if round(y_true_val,2) == round(y_pred_val,2):
            count+=1
            print(True)
        else:
            print(False)
            
    return count/n

compute_accuray(y_true=[0.113,1,2,3,4],y_pred=[0.2113,1,2,3,4])

compute_accuray(y_true=[2],y_pred=["4"])

# %%


#Euclidean Distance

def distance(p1: list[float], p2: list[float]):
    """This function computes the Euclidean distance between two
    vectors of equal length"""

    #make sure the lists are float not integers
    p1=[float(p1_item) for p1_item in p1]
    p2=[float(p2_item) for p2_item in p2]

    # print(p1,p2)
    #check for errors, empty lists, or non-equal lengths
    if not p1:
        raise ValueError("p1 and p2 vectors must not be empty")
    if not p2:
        raise ValueError("p1 and p2 vectors must not be empty")

    if len(p1) != len(p2):
        raise ValueError("p1 and p2 should have the same length")
    distance=0
    #do the euclidean geometry np.sqrt(np.sum(difference**2)

    for p1_itm,p2_itm in zip(p1,p2):
        distance += (p1_itm-p2_itm)**2

    return np.sqrt(distance)

distance(p1=[1,1,0],p2=[0,0,0])


# %%


def balance(data: list[float]): 
    """Let us check if the list is balanced"""
    if not data:
        raise ValueError("data can't be empty")

    #check if any number is not between 0 and1
    for i in data:
        if  0<= i <= 1:
            print(True, i)
        else:
            raise ValueError(f"data is not balanced. Number is {i}")
    print('data is balanced')
    return data


balance(data=[-0,1,1,0])

#%%

#OR 
def is_balanced_binary(data: list[int]):
    """Checks if a list of 0s and 1s is balanced (equal number of each)
    """

    if not data or not all(x in [0,1] for x in data):
        raise ValueError("Data must contain only 0s and 1s")

    count_0=data.count(0)
    count_1=data.count(1)

    return abs(count_0-count_1)<=1

is_balanced_binary([1, 0, 1, 1,0])  # True
# is_balanced_binary([1, 1, 1, 0])  # False

# %%


def add_ratio_feature(data: list[dict[str, float]]) -> list[dict[str, float]]:
    """
    For each dictionary with 'A' and 'B', add a new key 'A/B' = A divided by B.
    Handle division by zero by setting 'A/B' = 0.
    Example:
      [{'A': 10, 'B': 2}] → [{'A': 10, 'B': 2, 'A/B': 5.0}]
    """
    #initial chicken
    if not all(item for item in data.values()):
        raise ValueError("Data list should not be empty")
    data['A/B']=[]
    #Now let's loop over keys and add the ration
    for idx,i in enumerate(data['A']):
        # print(idx,i)
        if data['B'][idx] == 0:
            data['A/B'].append(0)
        else:
            data['A/B'].append((data['A'][idx]/data['B'][idx]))
        
    return data 


add_ratio_feature(data={'A':[10,23,3],'B':[2,0,4]})



# %%
#Or it was meant to be 

def add_ratio_feature(data:list[dict[str,float]]):

    if not data:
        raise ValueError("Data list should not be empty!")

    for entry in data:
        a=entry.get('A')
        b=entry.get('B')

        entry['A/B']= a/b if b!=0 else 0.0

    return data


data = [{'A': 10, 'B': 2}, {'A': 5, 'B': 0}, {'A': 9, 'B': 3}]
print(add_ratio_feature(data))

# %%

#smallest subarrays with maxmimum bitwise OR

def smallestSubarrays(nums):
    n = len(nums)
    answer = [0] * n
    last = [0] * 32  # last seen index for each bit

    for i in range(n - 1, -1, -1):
        # Update the last seen position for each bit
        for b in range(32):
            if nums[i] & (1 << b):
                last[b] = i
        
        # Find the farthest bit needed to match max OR
        farthest = i
        for b in range(32):
            farthest = max(farthest, last[b])
        
        answer[i] = farthest - i + 1

    return answer

smallestSubarrays([1, 0, 2, 1, 3])
         
# %%

def two_sums(nums: list[int], target: int) -> list[int]:
    """Returns the indices of the two numbers that add up to the target"""
    if not nums:
        raise ValueError("nums cannot be empty")

    num_map = {}  # Store value: index

    for idx, num in enumerate(nums):
        complement = target - num
        print(idx,num,complement)
        if complement in num_map:
            return [num_map[complement], idx]
        num_map[num] = idx

    return []  # No solution found

two_sums(nums=[2,7,3,4],target=9)


# %%
import numpy as np
def ispalindrome(x:int):

    """check if number is palindrome"""
    #the way to check is to get the number
    #start with small numbers up to 999
    third_place=np.floor(x/100)
    second_place=np.floor((x-100*third_place)/10)
    # print(second_place)
    first_place=x-third_place*100-10*second_place
    

    if third_place == first_place:
        return True
    else:
        return "Not Palindrome"

ispalindrome(333)

# %%