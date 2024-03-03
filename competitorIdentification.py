import pandas as pd
import numpy as np
import yfinance as yf
import yahooquery
import tensorflow_hub as hub
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from constants import AVAILABLE_SCREENS

class CompetitorAnalysis:

    def __init__(self, stock):
        """
        Initialises the class based on the requested company given as a command-line variable.
        self.ticker is a string representing the ticker symbol for the company.
        self.info saves all the basic information about the company from the yfinance api.
        """
        self.ticker = stock
        self.info = yf.Ticker(self.ticker).info

    def similar_group(self) -> str :
        """
        AVAILABLE_SCREENS is a constant that stores the categories that the available stocks are sorted into.
        In this method, we use the industry of our chosen company and work out which of the categories of stocks best fits the industry.
        To compute the 'best fit', we encode the screen names and industry name using the Google Universal Sentence Encoder as it captures the semantics of the sentence as well.
        The method then returns the stocks obtained from the resultant category.
        """
        screener = yahooquery.Screener()
        use_model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
        screeners_embedded = use_model(AVAILABLE_SCREENS)
        information = pd.DataFrame({'screeners':AVAILABLE_SCREENS, 'embeddings':np.array(x.numpy() for x in screeners_embedded)})
        industry_vector = use_model([self.info['industry']])[0].numpy()
        compute_similarity = lambda x: cosine_similarity([x], [industry_vector])[0][0]
        information['similarity'] = information['embeddings'].map(compute_similarity)
        maximum_similarity = max(information['similarity'])
        predicted_screen = information[information['similarity'] == maximum_similarity]
        return list(predicted_screen['screeners'])[0]

    def identify_market_competition(self) -> pd.DataFrame :
        """
        This method processes the result of the similar_group method.
        The list of stocks obtained from the similar_group method are all potential competitors to our chosen company.
        So, it collects all companies in a pandas dataframe and attaches the description of the business model of each to the dataframe.
        This dataframe is then the output of the method.
        """
        screener = yahooquery.Screener()
        most_likely_category = self.similar_group()
        potential_competition = screener.get_screeners(most_likely_category)[most_likely_category]['quotes']
        competitor_list = list(x['symbol'] for x in potential_competition)
        if self.ticker not in competitor_list:
            competitor_list.append(self.ticker)
        competitor_list = yahooquery.Ticker(competitor_list)
        data = pd.DataFrame(competitor_list.asset_profile).T
        relevent_data = pd.DataFrame(data, columns=['longBusinessSummary'])
        return relevent_data

    def prepare_clustering_data(self, paragraphs:pd.DataFrame) -> pd.DataFrame :
        """
        This method takes as input, the dataframe outputted by the self.identify_market_competition method.
        The next step is to process the description of the business model but any ML algorithm cannot process string-like data.
        So, this method uses the Google Universal Sentence Encoder to encode each description as a vector and attaches this to the dataframe.
        The method then outputs this modified dataframe.
        """
        paragraph_embedder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
        business_description_embedded = paragraph_embedder(paragraphs['longBusinessSummary'])
        paragraphs['summaryEncoded'] = list(x.numpy() for x in business_description_embedded)
        return paragraphs

    def kmeans_clustering(self, descriptions:pd.DataFrame, clusters:int) -> np.ndarray :
        """
        Takes the set of vectors representing the business model and runs a kmeans clustering algorithm to group together similar business descriptions.
        This method only runs the algorithm for a given number of clusters and this number is optimized in self.get_optimized_labels to maximise information gain.
        """
        kmeans_object = KMeans(n_clusters=clusters, random_state=1)
        kmeans_object.fit(np.array(list(descriptions['summaryEncoded'])))
        return kmeans_object.labels_

    def get_optimized_labels(self, descriptions:pd.DataFrame, optimisation_range:tuple) -> np.ndarray :
        """
        This method runs self.kmeans_clustering on a given range of cluster numbers and works out the optimum number to maximise the quality of clusters.
        This is done via maximising the silhouette score.
        The cluster associated with each description is then outputted.
        """
        k_values = range(optimisation_range[0], optimisation_range[1])
        max_silhouette_score = [0,0]
        for k in k_values:
            current_labels = self.kmeans_clustering(descriptions, k)
            silhouette_average = silhouette_score(np.array(list(descriptions['summaryEncoded'])), current_labels)
            if (silhouette_average > max_silhouette_score[0]):
                max_silhouette_score = [silhouette_average, current_labels]
        return max_silhouette_score[1]

    def obtain_competitors(self, descriptions:pd.DataFrame) -> list :
        """
        Runs the kmeans clustering algorithm on potential competitors and ones which are chosen into the same cluster are assumed to have the most similar business model.
        So, this methods groups together companies in the same cluster as the chosen one and this list is outputted.
        """
        if (len(descriptions) <= 5):
            return list(descriptions.index)
        max_clusters = int(len(descriptions) / 2) + 1
        optimum_labels = self.get_optimized_labels(descriptions, (2,max_clusters))
        descriptions['clusterLabel'] = optimum_labels
        select_cluster = int(descriptions.loc[self.ticker]['clusterLabel'])
        selected_competitors = list(descriptions[descriptions['clusterLabel'] == select_cluster].index)
        selected_competitors.remove(self.ticker)
        return selected_competitors

    def competitor_analysis_report(self):
        """
        Uses the result of self.obtain_competitors and writes a mini summary of results into a text file.
        """
        potential_competitors = self.identify_market_competition()
        revised_competitors = self.prepare_clustering_data(potential_competitors)
        final_companies = self.obtain_competitors(revised_competitors)
        ticker_list = yahooquery.Ticker(final_companies)
        information_dict = ticker_list.quote_type
        requested_company = yahooquery.Ticker([self.ticker]).quote_type[self.ticker]['shortName']
        file_name = f"competitionAnalysisReport_{self.ticker}.txt"
        if len(final_companies) > 0:
            write_string = f"Requested Company : {requested_company} ({self.ticker})\n\nPossible market competitors in current economic landscape based on business model :\n"
            count = 1
            for x in information_dict:
                write_string += f"\t{count}. {information_dict[x]['shortName']} ({x})\n"
                count += 1
        else:
            write_string = f"Requested Company : {requested_company} ({self.ticker})\n\nAlgorithm could not find another major company with noticably similar business models.\nLooks like the company chosen has exploited a gap in the market!"
        with open(file_name, "w") as f:
            f.write(write_string)

        

if __name__=="__main__":
    if len(sys.argv) != 2:
        raise Exception("Give only one ticker symbol in all caps")

    company = CompetitorAnalysis(sys.argv[1])
    company.competitor_analysis_report()