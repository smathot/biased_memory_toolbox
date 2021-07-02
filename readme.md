# Biased Memory Toolbox

*A Python toolbox for mixture modeling of data from visual-working-memory experiments*

Cherie Zhou (@cherieai) and Sebastiaan Mathôt (@smathot) <br />
Copyright 2020 - 2021


## Citation

Zhou, C., Lorist, M., Mathôt, S., (2021). Categorical bias in visual working memory: The effect of memory load and retention interval. *Cortex*. <https://osf.io/puq4v/>

*This manuscript is a Stage 1 in-principle acceptance of a registered report*


## Installation

```
pip install biased_memory_toolbox
```


## Usage

This section focuses on using the module, assuming that you have a basic understanding of mixture modeling of working memory data. If you want to know more about the theory behind mixture modeling, please read (for example) the manuscript cited above.

We start by reading in a data file using [DataMatrix](https://datamatrix.cogsci.nl/). The data should contain a column that contains the memoranda (here: `memory_hue`) and a column that contains the responses (here: `response_hue`), both in degrees with values between 0 and 360.



```python
from datamatrix import io

dm = io.readtxt('example-data/example-participant.csv')
```



As a first step, which is not related to mixture modeling per se, we check whether the participant performed significantly (p < .05) above chance. This is done with a permutation test that is implemented as `test_chance_performance()`. Here, low p-values indicate that performance deviates from chance.



```python
import biased_memory_toolbox as bmt

t, p = bmt.test_chance_performance(dm.memory_hue, dm.response_hue)
print('testing performance: t = {:.4f}, p = {:.4f}'.format(t, p))
```

__Output:__
``` .text
testing performance: t = -56.6385, p = 0.0000
```



Now let's fit the mixture model. We start with a basic model in which only the precision and the guess rate is estimated, as in the original [Zhang and Luck (2008)](https://doi.org/10.1038/nature06860) paper.

To do so, we first calculate the response error, which is simply the circular distance between the memory hue (the color that the participant needed to remember) and the response hue (the color that the participant reproduced). This is done with `response_bias()`, which, when no categories are provided, simply calculates the response error.



```python
dm.response_error = bmt.response_bias(dm.memory_hue, dm.response_hue)
```



We can fit the model with a simple call to `fix_mixture_model()`. By specifying `include_bias=False`, we fix the bias parameter (the mean of the distribution) at 0, and thus
only get two parameters: the precision and the guess rate.



```python
precision, guess_rate = bmt.fit_mixture_model(
    dm.response_error,
    include_bias=False
)
print('precision: {:.4f}, guess rate: {:.4f}'.format(precision, guess_rate))
```

__Output:__
``` .text
precision: 1721.6386, guess rate: 0.0627
```



Now let's fit a slightly more complex model that also includes a bias parameter. To do so, we first calculate the response 'bias', which is similar to the response error except that it is recoded such that positive values reflect a response error towards the prototype of the category that the memorandum belongs to. For example, if the participant saw a slightly aqua-ish shade of green but reproduced a pure green, then this would correspond to a positive response bias for that response.

To calculate the response bias we need to specify a `dict` with category boundaries and prototypes when calling `response_bias()`. A sensible default (`DEFAULT_CATEGORIES`), based on ratings of human participants, is provided with the toolbox.



```python
dm.response_bias = bmt.response_bias(
    dm.memory_hue,
    dm.response_hue,
    categories=bmt.DEFAULT_CATEGORIES
)
```



Next we fit the model again by calling `fit_mixture_model()`. We now also get a bias parameter (because we did not specify `include_bias=False`) as described in [Zhou, Lorist, and Mathôt (2021)](https://osf.io/puq4v/).



```python
precision, guess_rate, bias = bmt.fit_mixture_model(dm.response_bias)
print(
    'precision: {:.4f}, guess rate: {:.4f}, bias: {:.4f}'.format(
        precision,
        guess_rate,
        bias
    )
)
```

__Output:__
``` .text
precision: 1725.9568, guess rate: 0.0626, bias: 0.5481
```



It also makes sense to visualize the model fit, to see if the model accurately captures the pattern of responses. We can do this by plotting a probability density function, which can be generated by `mixture_model_pdf()`.

```
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x = np.linspace(-180, 180, 360)
y = bmt.mixture_model_pdf(x, precision, guess_rate, bias)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Model fit')
plt.xlim(-50, 50)
plt.plot(x, y)
plt.subplot(122)
plt.title('Histogram of response biases')
plt.xlim(-50, 50)
sns.distplot(dm.response_bias, kde=False)
plt.savefig('example.png')
```

![](https://github.com/smathot/biased_memory_toolbox/raw/master/example.png)

We can also fit a model that takes into account swap errors, as described by [Bays, Catalao, and Husain (2009)](https://doi.org/10.1167/9.10.7). To do so, we need to also specify the response bias (or plain error) with respect to the non-target items.

Here, we select only those trials in which the set size was 3, and then create two new columns for the response bias with respect to the second and third memory colors, which were non-targets in this experiment. (The first color was the target color.)



```python
dm3 = dm.set_size == 3
dm3.response_bias_nontarget2 = bmt.response_bias(
    dm3.hue2,
    dm3.response_hue,
    categories=bmt.DEFAULT_CATEGORIES
)
dm3.response_bias_nontarget3 = bmt.response_bias(
    dm3.hue3,
    dm3.response_hue,
    categories=bmt.DEFAULT_CATEGORIES
)
```



By passing a list of non-target response biases, we get a fourth parameter: swap rate.



```python
precision, guess_rate, bias, swap_rate = bmt.fit_mixture_model(
    x=dm3.response_bias,
    x_nontargets=[
        dm3.response_bias_nontarget2,
        dm3.response_bias_nontarget3
    ],
)
print(
    'precision: {:.4f}, guess rate: {:.4f}, bias: {:.4f}, swap_rate: {:.4f}'.format(
        precision,
        guess_rate,
        bias,
        swap_rate
    )
)
```

__Output:__
``` .text
precision: 1458.9628, guess rate: 0.0502, bias: 1.2271, swap_rate: 0.0191
```




## License

`biased_memory_toolbox` is licensed under the [GNU General Public License
v3](http://www.gnu.org/licenses/gpl-3.0.en.html).
