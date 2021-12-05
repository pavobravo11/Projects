"""
Created on Sunday November 28

@author: Gustavo Bravo
"""

import pandas as pd
import numpy as np
import plotly.express as px
import os


# This function will get all the data found in the Adj Close Column for every file in the passed directory
# params: path = path of the folder in which you want the data from
# returns: data_list[], all the data compiled onto one list
def get_data(path, names_path):
    data_dict = {}
    list_names = pd.read_csv(
        r"" + names_path).to_numpy()

    for filename in os.listdir(r"" + path):
        company_name = list_names[np.where(list_names == filename.strip(".csv"))[0][0], 1]
        data_dict[company_name] = pd.read_csv(os.path.join(path, filename))["Adj Close"].to_list()
    return data_dict


# This function returns a dictionary with the monthly returns of each stock
# params: data in the form of a dictionary
# returns: dictionary of returns
def yield_returns(data_dict):
    # Creates dictionary to store yearly returns
    returns_dict = {}
    for i in data_dict:
        temp = data_dict[i]
        # The first return will be 0
        returns = []
        # Calculate the return for all else
        for j in range(1, len(temp)):
            returns.append(temp[j] / temp[j - 1] - 1)
        returns_dict[i] = returns
    return returns_dict


# This gets the average yearly returns for a specified period of time
# params: return_dict, a dictionary storing the monthly returns
#           start_date, beginning date for averages
#           end_date, end date for averages
# returns a tuple of dictionaries
def get_yearly_return_and_volatility(returns_dict, start_date, end_date):
    avg_dict = {}
    vol_dict = {}
    for i in returns_dict:
        avg_dict[i] = np.average(returns_dict[i][start_date: end_date]) * 12
        vol_dict[i] = np.std(returns_dict[i][start_date: end_date], ddof=1) * np.sqrt(12)
    return avg_dict, vol_dict


# Create a portfolio of the stocks blended at a certain weights of stocks
# params: weights[], list of 4 weights that add up to 1
#       returns_list{}, dictionary that includes the weights
#       cov_matrix, the covariance matrix of the 4 stocks
# returns: void
def create_portfolio(weights, returns_list, cov_matrix, plot_returns, plot_volatility, legend):
    if round(weights.sum()) != 1:
        raise ValueError("Weights must add up to 1")
    c_matrix = cov_matrix.copy()

    # Calculate expected return for portfolio
    expected_returns = weights[0] * returns_list[0] + \
        weights[1] * returns_list[1] + \
        weights[2] * returns_list[2] + \
        weights[3] * returns_list[3]

    # Add them to our global data
    plot_returns.append(expected_returns)

    # Calculate volatility for portfolio using the covariance matrix
    for idx, name in enumerate(c_matrix):
        c_matrix.loc[name, :] *= weights[idx]
        c_matrix.loc[:, name] *= weights[idx]

    # Add Volatility to global data
    plot_volatility.append(np.sqrt(c_matrix.sum().sum() * 12))

    # Give portfolio a name based off the weights and add it to global data
    name = ""
    legend.append(name.join(str(weights)))

    return plot_returns, plot_volatility, legend


# randomly generate n number of portfolios
def generate_random_portfolios(n, returns, cov_matrix, returns_ledger, volatility_ledger, legend):
    for i in range(n):
        returns_ledger, volatility_ledger, legend = create_portfolio(
            weights=np.random.dirichlet(np.ones(4), size=1)[0],
            returns_list=returns,
            cov_matrix=cov_matrix,
            plot_returns=returns_ledger,
            plot_volatility=volatility_ledger,
            legend=legend)
    return returns_ledger, volatility_ledger, legend


# This function creates the covariance matrix for a specified period of time from a dictionary
def create_covariance_matrix(keys, values, start_date, end_date):
    cov_matrix = pd.DataFrame(data={keys[0]: values[0][start_date:end_date],
                                    keys[1]: values[1][start_date:end_date],
                                    keys[2]: values[2][start_date:end_date],
                                    keys[3]: values[3][start_date:end_date]}).cov()
    return cov_matrix


# This function plots the given data
def plot(names, returns, volatility, title):
    df = pd.DataFrame(data=None, columns=["Names", "Average Annual Returns", "Volatility of Returns"])
    df.loc[:, "Names"] = names
    df.loc[:, "Average Annual Returns"] = returns
    df.loc[:, "Volatility of Returns"] = volatility

    fig = px.scatter(
        data_frame=df,
        x="Volatility of Returns",
        y="Average Annual Returns",
        color="Names",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        opacity=0.5,
        title=title)

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    fig.update_traces(marker=dict(size=15,
                                  line=dict(width=1,
                                            color='Black')),
                      selector=dict(mode='markers'))

    fig.show()


# This method is a parent method that finds average returns and volatility of a stock and then generates
# 15 randomly blended portfolios based on those dates of data
def train_data(monthly_returns, start_date, end_date):
    average_return, average_volatility = get_yearly_return_and_volatility(returns_dict=monthly_returns,
                                                                          start_date=start_date,
                                                                          end_date=end_date)
    returns = list(average_return.values())
    volatility = list(average_volatility.values())
    legend = list(average_return)

    cov_matrix = create_covariance_matrix(keys=list(monthly_returns),
                                          values=list(monthly_returns.values()),
                                          start_date=start_date,
                                          end_date=end_date)

    returns, volatility, legend = generate_random_portfolios(
        n=15,
        returns=list(average_return.values()),
        cov_matrix=cov_matrix,
        returns_ledger=returns,
        volatility_ledger=volatility,
        legend=legend
    )

    return returns, volatility, legend


def main():
    # Get the data
    data = get_data(path="C:\\Users\\gusta\\Python\\Projects\\School\\Stock Data",
                    names_path="C:\\Users\\gusta\\Python\\Projects\\School\\Stock Names\\Names.csv")

    # Find the returns
    monthly_returns = yield_returns(data_dict=data)

    returns13, volatility13, legend13 = train_data(monthly_returns=monthly_returns,
                                                   start_date=0,
                                                   end_date=36)

    # Plotly!!
    plot(names=legend13,
         returns=returns13,
         volatility=volatility13,
         title="2017 - 2019 Risk and Return")


if __name__ == "__main__":
    main()
