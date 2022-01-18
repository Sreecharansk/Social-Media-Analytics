"""
Social Media Analytics Project
Name:
Roll Number:
"""

from os import listdir
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    #filename=filename.csv
    file_df = pd.read_csv(filename)
    return file_df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    start = fromString.find("From: ") + len("From: ")
    line = fromString[start:]
    end = line.find(" (")
    line = line[:end]
    line = line.strip()
    return line


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    start = fromString.find(" (") + len(" (")
    line = fromString[start:]
    end = line.find("from")
    line = line[:end]
    line = line.strip()
    return line


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    start = fromString.find(" from") + len(" from")
    line = fromString[start:]
    end = line.find(")")
    line = line[:end]
    line = line.strip()
    return line


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    tags=message.split("#")
    K=[' ', '\n', '#', '.', ',', '?', '!', ':', ';', ')']
    val=""
    list=[]
    for i in tags[1:]:
        for j in i:
            if j not in K:
                val=val+j
            else:
                break
        list.append('#'+val)
        val=""
    return list


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    return(stateDf.loc[stateDf['state'] == state, 'region'].iloc[0])


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    name=[]
    position=[]
    state=[]
    hashtags=[]
    region=[]
    for i in range(len(data)):
        st=data['label'][i]
        A=parseName(st)
        name.append(A)
        B=parsePosition(st)
        position.append(B)
        C=parseState(st)
        state.append(C)
        st1=data['text'][i]
        D=findHashtags(st1)
        hashtags.append(D)

    for i in range(len(state)):
        E=getRegionFromState(stateDf,state[i])
        region.append(E)

    data['name']=name
    data['position']=position
    data['state']=state
    data['region']=region
    data['hashtags']=hashtags
    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"
    


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiment=[]
    for i in range(len(data)):
        st=data['text'][i]
        A=findSentiment(classifier, st)
        sentiment.append(A)      
    data['sentiment'] = sentiment
    return
    
        


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    st1=[]
    st2=[]
    dtc=[]
    dd=[]
    if colName!="" and dataToCount!="":
        for i in range(len(data)):
            st=data['state'][i]
            st1.append(st)
            dt=data[colName][i]
            dtc.append(dt)
            if st not in st2:
                st2.append(st) #all unique names of states
        for i in range(len(st2)):
            A=st2[i]
            count=0
            for j in range(len(st1)):
                if A==st1[j]:
                    if dtc[j]==dataToCount:
                        count=count+1
            dd.append(count)            
        sta=[]
        cnt=[]
        for i in range(len(dd)):
            if dd[i]!=0:
                sta.append(st2[i])
                cnt.append(dd[i])
        dic = {}
        for i in range(0,len(sta)):
            dic[sta[i]] = cnt[i]
        return dic
    else:
        for i in range(len(data)):
            st=data['state'][i]
            st1.append(st)
            if st not in st2:
                st2.append(st) #all unique names of states
        for i in range(len(st2)):
            A=st2[i]
            count=0
            for j in range(len(st1)):
                if A==st1[j]:
                    count=count+1
            dd.append(count)
        sta=[]
        cnt=[]
        for i in range(len(dd)):
            if dd[i]!=0:
                sta.append(st2[i])
                cnt.append(dd[i])
        dic = {}
        for i in range(0,len(sta)):
            dic[sta[i]] = cnt[i]
        return dic

'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    reg=[]
    cln1=[]
    cln2=[]
    reg2=[]
    d1={}
    for i in range(len(data)):
            re=data['region'][i]
            reg.append(re)
            if re not in reg2:
                reg2.append(re) #all unique names of regions           
            cln=data[colName][i]
            cln1.append(cln)
            if cln not in cln2:
                cln2.append(cln) #all unique names of colname attributes
    for i in range(len(reg2)):
        d1[reg2[i]]={}
        for j in range(len(cln2)):
            d1[reg2[i]][cln2[j]]=0
    
    for i in range(len(reg2)):
        A=reg2[i]
        for j in range(len(cln2)):
            B=cln2[j]
            count=0
            for k in range(len(data)):
                C=data['region'][k]
                D=data[colName][k]
                if A==C and B==D:
                    count=count+1
            d1[A][B] = count
       
    return d1


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    dict={}
    for i in data["hashtags"]:
        for j in i:
            if len(j)!=0 and j not in dict:
                dict[j]=1
            else:
                dict[j]=dict[j]+1
    return dict


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    dic={}
    A = list(hashtags.keys())
    B = list(hashtags.values())
    for i in range(count):
        A1=max(B)
        index = B.index(A1)
        dic[A[index]]=A1
        B[index]=0
    return dic


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    htgs=[]
    senti=[]
    ind=[]
    for i in range(len(data)):
        a=data['hashtags'][i]
        htgs.append(a)
        b=data['sentiment'][i]
        senti.append(b)
    count=0
    for i in range(len(htgs)):
        if hashtag in htgs[i]:
            ind.append(i)
    for i in range(len(ind)):
        A=senti[ind[i]]
        if A=="positive":
            count=count+1
        if A=="negative":
            count=count-1
        if A=="neutral":
            count=count+0                
    return (count/len(ind))


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    A = list(stateCounts.keys())
    B = list(stateCounts.values())
    plt.bar(A,B)
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation = 90)
    plt.show()

    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    B = list(stateFeatureCounts.keys())
    dict={}
    for i in range(len(B)):
        dict[B[i]]= stateFeatureCounts[B[i]]/stateCounts[B[i]]
    dic={}    
    A = list(dict.keys())
    B = list(dict.values())
    for i in range(n):
        A1=max(B)
        index = B.index(A1)
        dic[A[index]]=A1
        B[index]=0
    plt.bar(list(dic.keys()),list(dic.values()))
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Feature Rate')
    plt.xticks(rotation = 90)
    plt.show()
   
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    A = list(regionDicts.keys())
    B = list(regionDicts.values())
    D=list(B[0].keys()) ## x labels 
    South=list(B[0].values())
    West=list(B[1].values())
    Midwest=list(B[2].values())
    Northeast=list(B[3].values())
    w=0.2
    bar1=[]
    for i in range(len(D)):
        bar1.append(i)
    bar2=[i+w for i in bar1]
    bar3=[i+w for i in bar2]
    bar4=[i+w for i in bar3]

    plt.bar(bar1,South,w,label="South")
    plt.bar(bar2,West,w,label="West")
    plt.bar(bar3,Midwest,w,label="Mid West")
    plt.bar(bar4,Northeast,w,label="North East")

    plt.xlabel("Feature")
    plt.ylabel("value")
    plt.title(title)
    plt.xticks(bar3,D)
    plt.legend()
    plt.show()
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    p=getHashtagRates(data)
    top=mostCommonHashtags(p,50)
    hash=[]
    freq=[]
    senti=[]
    for i in top:
        hash.append(i)
        freq.append(top[i])
        senti.append(getHashtagSentiment(data, i))
    scatterPlot(freq,senti,hash,"Top 50 Hastags v/s Sentiment")
    
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()