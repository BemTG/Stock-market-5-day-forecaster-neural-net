# Stock-market-5-day-forecaster-neural-net


     
Many investment banks and hedge funds around the world trade stocks to create wealth for their investors. When it comes to trading the main goal is to preserve wealth and manage risk. One of the many ways hedge funds execute trades is through quantitative research and analysis.<br><br>
    <h4><strong style="color: white"> Staying market neutral</strong></h4>
    <p style="color: white">Quantitative traders use data to make predictions. This is usually carried out by making simple models to capture the markets state and forecast trends based on historical patterns. Quants find patterns in these historical datasets and make risk-adjusted statistical bets based on the patterns and behaviour of the market.<br>
A simple model such as the Capital Asset Pricing Model (CAPM) can be used to describe the relationship between systematic risk and expected return of a stock to help execute a trade.
</p>

<math>
  <h2 style="color: white ; text-align:center"><strong>r =  βr(t) + α(t)</strong></h2>
  <p style="color: white ; text-align:left"> <strong>r</strong>= Total returns</p>
  <p style="color: white ; text-align:left"> <strong>βr(t)</strong>= stock return attributed to the market</p>
  <p style="color: white ; text-align:left"> <strong>β</strong>= market correlation</p>
  <p style="color: white ; text-align:left"> <strong>α</strong>= alpha (residual)</p>
</math><br><br>




    <!-- <strong> Which economic reports are being used?</strong> -->
    <p style="color: white">The total return of a stock can be explained using the two variables, market return and residual return. β is simply a measure of how correlated the given stock returns are to the market whereas α is the residual returns attributed to the stock irrespective of the market returns.<br>
Quant traders build strategies and models to predict the alpha factor of the given stock to take market neutral trades. <br> 
Hedge funds will use a combination of different models (including machine learning models) to predict if the given stock is going to be above or below the market by a certain timeframe. The objective is to take trades on both sides of the market (long and short) to limit exposure to the fluctuation of the overall market direction.<br><br>

A simple example of a market-neutral trade:<br><br>

<math>
  <h2 style="color: white ; text-align:center"><strong>r(p)=  W <sub>A</sub>  β<sub>A</sub> + W <sub>B</sub>  β<sub>B</sub>  +   W <sub>A</sub>  α<sub>A</sub> + W <sub>B</sub>  α<sub>B</sub>  </strong></h2><br>
  <p style="color: white ; text-align:left"> <strong>W <sub>A</sub> </strong>= Weight allocation of stock A </p>
  <p style="color: white ; text-align:left"> <strong>W <sub>B</sub> </strong>= Weight allocation of stock B</p>
  <p style="color: white ; text-align:left"> <strong> β<sub>A</sub></strong>= market correlation of stock A</p>
   <p style="color: white ; text-align:left"> <strong> β<sub>A</sub></strong>= market correlation of stock B</p>
    <p style="color: white ; text-align:left"> <strong> α<sub>A</sub></strong>= Returns of stock A irrespective of the market (residual)</p>
    <p style="color: white ; text-align:left"> <strong> α<sub>B</sub></strong>= Returns of stock B irrespective of the market (residual)</p>

</math><br><br>

 <p style="color: white">Let assume our predictive model is forecasting that stock A is going to increase by 1% above the market. To take a neutral trade we would have to find a stock that is inversely correlated to the market. As a result, we find that stock B will be trading -0.5% below the market. Since we cannot change the β values of the stock we would have to adjust the weighting (W A & W B) accordingly to reduce market exposure to zero in the first two variables. For this specific example if stock A has βA=1 and stock B is βB=-1 , then W B would have to be 4 whenever W A is 2. Consequently, the expected return r(p) is going to derive solely from the last 2 variables in equation (W A αA + W B αB) and hence will be neutral from the overall market returns. The trade will only be successful if the alpha factors are correct that was modelled.</p><br>

 <p style="color: white">It is important to realise that hedge funds spend a substantial amount of money constructing, developing and improving the predictive models since inaccuracy in the model can lead to a considerable amount of losses. </p><br><br>

 <h4><strong style="color: white"> How does the model in the webapp work?</strong></h4><br>
    <p style="color: white">TThe model behind the web app is a neural net supervised machine learning model, which has been trained with 5 years of historical data of all the SP500 companies. The feature sets it was trained on are the daily  Open, High, Low, Adjusted Close values. By receiving the daily feature sets  (from yahoo finance) the model forecasts the upcoming 5th-day value to indicate the trend of the stock.<br><br>
      Back-propagation is the essence of neural net training. It is the practice of fine-tuning the weights of a neural net based on the error rate to capture the patterns and relationships of the data set. The user does not necessarily need to specify what patterns to look for. Once the model architecture has been constructed and the learning parameters have been adjusted, the neural network learns on its own.<br><br>
      Since our data is relatively small we used a simple model architecture of 2 hidden layers with 5 dense nodes. The activation function we have used is relu (rectified linear unit). By stacking several of these dense nodes we can create a higher order of polynomials which help capture more complex patterns of the dataset. <br><br>
      The data was split 67% for training and 33% for testing. We trained our small dataset for 3 epochs for which we measured the mean squared error to be ~19%.<br><br>

      For more information about the project and the source code visit <a href="https://github.com/BemTG/Stock-market-forecasting-neural-net-and-XGBoost-models-/blob/master/Stock%20market%20forecasting%20using%20sequential%20neural%20net%20model.ipynb">github </a> link.</p><br><br>

      <h4><strong style="color: white"> Improvements</strong></h4><br>
    <p style="color: white">There is certainly a lot of room for improvement, one of which is to train the model for longer (only trained model for 5-10 mins) to reduce the loss further.<br><br>

      In addition, we can add more market features to the dataset including RSI, Bollinger bands and moving averages data when training the model. This may decrease the loss and improve accuracy as it can help the model learn faster and effectively.<br><br> Moreover, we are not retraining the model overtime hence model can diverge from optimal weights as new data points come in.<br><br>













  </p>


     </p>


     </p>
    <p></p>
  </div>
</div>
