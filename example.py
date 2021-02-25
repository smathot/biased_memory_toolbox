"""
# Biased Memory Toolbox (example)

First import all relevant libraries.
"""

import biased_memory_toolbox as bmt
from datamatrix import io, operations as ops, DataMatrix
import numpy as np
from matplotlib import pyplot as plt


"""
Read in a data file as a DataMatrix. There should be a column that contains
the memoranda (here: `hue1`) and a column that contains the responses (here:
`hue_response`), both in degrees with values between 0 and 360.

Next, we use the `response_bias()` function from `biased_memory_toolbox`
(imported as `bmt`) to calculate the `response bias`, which is the response
error in the direction of the category prototype. We use the default categories
as defined in the toolbox.
"""

dm = io.readtxt('data_category.csv')
dm.response_bias = bmt.response_bias(
    dm.hue1,
    dm.hue_response,
    bmt.DEFAULT_CATEGORIES
)


"""
Next, we loop through all participants, and for each participant separately
fit a mixture model to `response_bias`, resulting in a precision, guess rate,
and bias. This is done with `bmt.fit_mixture_model()`.

We also check whether participants responded above chance. This is done by
`bmt.test_chance_performance()`.
"""

# Initialize an empty DataMatrix that will contain the fit results for each
# participant.
sm = DataMatrix(dm.sessionid.count)
sm.precision = -1
sm.guess_rate = -1
sm.bias = -1
sm.sessionid = -1
sm.p_chance = -1
# Intialize a plot that will contain the fits for individual participants.
plt.xlim(-75, 75)
plt.xlabel('Error towards prototype (deg)')
plt.ylabel('Probability')
plt.axvline(0, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
# Split the DataMatrix (dm) based on session id, such that we can loop through
# the for each participant separately.
for row, (sessionid, sdm) in zip(sm, ops.split(dm.sessionid)):
    row.sessionid = sessionid
    # Fit the mixture model and assign the parameters to the row of `sm`.
    row.precision, row.guess_rate, row.bias = \
        bmt.fit_mixture_model(sdm.response_bias)
    # Test chance performance. The first return value is the t value, which we
    # don't use. The second return value is the p value, which we assign to
    # the row of `sm`.
    _, row.p_chance = bmt.test_chance_performance(
        sdm.hue1,
        sdm.hue_response    
    )
    x = np.linspace(-180, 180, 100)
    y = bmt.mixture_model_pdf(x, row.precision, row.guess_rate, row.bias)
    plt.plot(x, y)
io.writetxt(sm, 'mixture-model-results.csv')


# % output
# ![](/home/sebastiaan/.opensesame/image_annotations/4d3966a73c204282913f55adc52efb63.png)
# 
"""
Show the results!
"""

sm

# % output
# DataMatrix[36, 0x7fc41f496730]
# +----+---------------------+----------------------+------------------------+--------------------+--------------+
# | #  |         bias        |      guess_rate      |        p_chance        |     precision      |  sessionid   |
# +----+---------------------+----------------------+------------------------+--------------------+--------------+
# | 0  |  1.4217392979155254 | 0.052181652199398444 | 7.712989825379358e-33  |    1.189784E+03    | 1.585730E+12 |
# | 1  |  5.667487731518942  |          0           | 6.682329322041846e-25  | 436.87060009180897 | 1.585737E+12 |
# | 2  |  1.1722770078213238 |          0           | 4.037222269463366e-31  | 653.0757954350673  | 1.585743E+12 |
# | 3  |  3.7706330429912907 | 0.009072369172030186 | 4.1077279016054385e-31 | 923.5872042348702  | 1.585914E+12 |
# | 4  |  -2.391284358866572 | 0.038175225972556556 | 4.902083750802149e-30  | 981.2399238539302  | 1.586164E+12 |
# | 5  |   5.18644033164451  |          0           |  2.0304760910699e-26   | 470.90514844721145 | 1.586167E+12 |
# | 6  |  1.4726297313538856 |          0           | 7.920251880215398e-09  | 172.19032115822907 | 1.586195E+12 |
# | 7  | -1.0921689646710373 |  0.1346907163338175  |  7.15932302760606e-32  |    1.818878E+03    | 1.586255E+12 |
# | 8  |  -1.155818946015223 | 0.04821900929782717  | 1.169355644570441e-31  | 697.7330985207599  | 1.586266E+12 |
# | 9  | -0.2238589655418146 |          0           | 1.507204942804091e-29  | 651.5972843831914  | 1.586516E+12 |
# | 10 |  6.316882845226377  |          0           | 1.1368128642528372e-22 | 391.34924016604515 | 1.586518E+12 |
# | 11 |  5.137531756034952  |          0           |  0.000278415942128514  | 646.7724041950485  | 1.586521E+12 |
# | 12 | -0.7038535694070114 |          0           | 3.6944382849072854e-27 |    1.000225E+03    | 1.586771E+12 |
# | 13 |  2.645177355899998  | 0.02015808313987081  | 2.1213355529775972e-27 | 903.6773466573233  | 1.586792E+12 |
# | 14 |  17.078842340504718 |          0           | 2.8482228252226376e-11 | 211.52873184882478 | 1.586854E+12 |
# | 15 |  2.344974643339937  |          0           | 2.580918861038652e-29  | 675.9897320420317  | 1.586861E+12 |
# | 16 |  0.7214083757289214 | 0.019613334578056092 | 2.6417189471777844e-26 | 799.5859163201269  | 1.586866E+12 |
# | 17 |  1.8603145152040366 |          0           | 2.793392672464018e-06  |    1.242530E+03    | 1.587908E+12 |
# | 18 |  1.2722497979874146 | 0.037075632895014164 | 8.597519313696515e-26  | 500.0939818023424  | 1.589097E+12 |
# | 19 |  1.950781415499978  |          0           | 1.4208637558480825e-32 |  379.505757763179  | 1.589358E+12 |
# +----+---------------------+----------------------+------------------------+--------------------+--------------+
# (+ 2 rows not shown)
# 
