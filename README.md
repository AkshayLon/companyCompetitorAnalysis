# Project Description
<p>
  Imagine that you are an investor and you are researching any general company. A part of the process involves looking into rival firms and how the competition might play out in the market.
  Identifying which firms are directly in competition with the one you are analysing is not as simple as picking companies in the same industry or sector. There is also a need to look deeper into the business model and analyse factors such as the nature of the customer base etc.
  Well this program does just that.
</p>
<p>
  This program uses NLP techniques as well as other machine learning algorithms to analyse various business models at a deeper level and display which companies are most likely to be in direct competition with a given firm that you are analysing. This will hopefully make your investment research process more efficient and accurate.
</p>

# How to use the code (Example use case)
Firstly, install the required packages (open a terminal and execute the following command) :
```bash
$ pip install requirements.txt
```
When running the script, the ticker symbol of the company that you want to analyse is given as a command-line argument. Here, we will use Nvidia (NVDA) as an example. 
Once in the folder of the program, type into the terminal :
```bash
$ python .\competitorIdentification.py NVDA
```
The "NVDA" can be replaced with the ticker symbol of any US company. In this case, after the program is run, it will create a text file called "competitionAnalysisReport_NVDA.txt" with the contents as follows :
```txt
Requested Company : NVIDIA Corporation (NVDA)

Possible market competitors in current economic landscape based on business model :
	1. Advanced Micro Devices, Inc. (AMD)
	2. Intel Corporation (INTC)
	3. QUALCOMM Incorporated (QCOM)
	4. Micron Technology, Inc. (MU)
	5. Monolithic Power Systems, Inc. (MPWR)
```
This output file can now be used in your own analysis or as a part of a wider pipeline as you wish!
