# coding=utf-8


# <markdowncell>
"""
# Biased Memory Toolbox (example)

First import all relevant libraries.
"""
# </markdowncell>


# <codecell>
import biased_memory_toolbox as bmt
from datamatrix import io, operations as ops, DataMatrix
import numpy as np
from matplotlib import pyplot as plt
# </codecell>


# <markdowncell>
"""
Read in a data file as a DataMatrix. There should be a column that contains
the memoranda (here: `hue1`) and a column that contains the responses (here:
`hue_response`), both in degrees with values between 0 and 360.

Next, we use the `response_bias()` function from `biased_memory_toolbox`
(imported as `bmt`) to calculate the `response bias`, which is the response
error in the direction of the category prototype. We use the default categories
as defined in the toolbox.
"""
# </markdowncell>


# <codecell>
dm = io.readtxt('data_category.csv')
dm.response_bias = bmt.response_bias(
    dm.hue1,
    dm.hue_response,
    bmt.DEFAULT_CATEGORIES
)
# </codecell>


# <markdowncell>
"""
Next, we loop through all participants, and for each participant separately
fit a mixture model to `response_bias`, resulting in a precision, guess rate,
and bias. This is done with `bmt.fit_mixture_model()`.

We also check whether participants responded above chance. This is done by
`bmt.test_chance_performance()`.
"""
# </markdowncell>


# <codecell>
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
# </codecell>


# <markdowncell>
"""
Show the results!
"""
# </markdowncell>


# <codecell>
sm
# </codecell>
