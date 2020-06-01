import ast
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import collections

def read_result(FilePath = "result.txt"):
    with open(FilePath,'r') as file:
        lines = file.readlines()
    lines = [line.strip("\n") for line in lines]
    resultDict ={}
    for i in range(0,len(lines),2):
        resultDict[float(lines[i].split("_")[-1])*100] = ast.literal_eval(lines[i+1])
    resultDict = collections.OrderedDict(sorted(resultDict.items()))
    return resultDict
def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.00*height,height,ha='center', va='bottom')
def plot(resultDict):
    x = list(resultDict.keys())
    values = [round(vl['corr'],2) for vl in resultDict.values()]
    x_pos = [i for i, _ in enumerate(x)]
    fig, ax = plt.subplots()
    rects = ax.bar(x_pos, values, color='green')
    plt.xlabel("% error")
    plt.ylabel("Pearson-Spearman corr")
    plt.title("Results on STS-B Dataset")
    plt.xticks(x_pos, x)
    autolabel(ax,rects)
    plt.show()
    fig.savefig("sts-b_50eps.png")

def batch_result(dataset_name="imdb",result="mcc"):
    p_directory = "output/"
    resultDict = {}
    for directory in os.listdir(p_directory):
        if not directory.startswith('.') and directory.startswith(dataset_name):
            with open(p_directory+directory+"/eval_results.txt") as fp:
                lines = fp.readlines()
            lines = [line.strip("\n") for line in lines]
            data = []
            for line in lines:
                if line.startswith(result):
                    data.append(round(float(line.split("=")[-1]),3))
            resultDict[float(directory.split("_")[-1])*100] = data
    resultDict = collections.OrderedDict(sorted(resultDict.items()))
    return resultDict

def line_plot(resultDict):
    for key,item in resultDict.items():
        plt.plot(item, label=str(key)+"% error")
    # Add labels and title
    plt.title("Cross Validation error after every 50 steps on IMDB dataset ")
    plt.xlabel("steps")
    plt.ylabel("F1-Score")

    plt.legend()
    plt.savefig("f3.png")
            
            
            
            
            
            


#resultDict = batch_result()
#line_plot(resultDict)
resultDict = read_result("sts-b_result_50eps.txt")
plot(resultDict)