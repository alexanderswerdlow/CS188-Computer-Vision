# %%
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
session = WolframLanguageSession()

# %%
session.evaluate(wl.Integrate(wlexpr('x^2'), wlexpr('x')))


# %%
