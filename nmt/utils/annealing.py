# Source: https://github.com/wilkeraziz/dgm4nlp
class AnnealingSchedule:
  """
  This class implements helper code for an annealing schedule.
  """

  def __init__(self, initial=1., final=1., step=0.,
               wait=0, nb_updates=1, wait_value=None,
               step_fn=lambda alpha, step, final: min(alpha + step, final),
               name='alpha'):
    """
    :param initial:
    :param final:
    :param step:
    :param wait: how many updates should we wait before starting the schedule (the first step occurs after wait + nb_updates)
    :param nb_updates: number of updates between steps
    :param wait_value: a value (other than initial) to use while waiting
    :param step_fn: control a step in the schedule
        - the default step is additive and capped by `final`
        - one can design multiplicative steps
        - once can even make it a decreasing schedule
    :param name:
    """
    self._initial = initial
    self._final = final
    self._step = step
    self._wait = wait
    self._nb_updates = nb_updates
    self._alpha = initial
    self._alpha_while_waiting = wait_value if wait_value is not None else initial
    self._step_fn = step_fn
    self._counter = 0
    self.name = name

  def alpha(self):
    """Return the current alpha"""
    if self._wait > 0:
      return self._alpha_while_waiting
    return self._alpha

  def update(self):
    """
    Update schedule or waiting time.
    :param eoe: End-Of-Epoch flag
    :return: current alpha
    """
    if self._wait > 0:    # we are still waiting
      self._wait -= 1     # decrease the waiting time and keep waiting
      return self._alpha_while_waiting  # and keep waiting
    # We are done waiting, now we start counting
    self._counter += 1
    if self._counter < self._nb_updates:  # not enough updates
      return self._alpha
    else:  # enough updates, we are ready to reset the counter to zero 
      self._counter = 0
    # and apply a step 
    self._alpha = self._step_fn(self._alpha, self._step, self._final)
    return self._alpha
