from __future__ import division

import numpy as np
from abc import ABCMeta, abstractmethod

import scipy
from scipy.stats import truncexpon
from itertools import chain, combinations, permutations

# abstract class python 2 & 3 compatible
ABC = ABCMeta('ABC', (object,), {})


def create_eyeblink_trials(N_batch, input_length, t_stim, t_delay):
    0

def create_flipflop_trials(N_batch, input_length, t_stim, t_delay, input_amp=1., target_amp=0.5,):
    """
    Creates N_batch trials of the flip-flop task.
    During each trial, the network receives a number of short pulses of duration t_stim.
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask, 0
    -------
    """
    inputs = np.zeros((N_batch, input_length, 2))
    outputs = np.zeros((N_batch, input_length, 2))

    for i in range(N_batch):
        pulses = [0]    
        while pulses[-1] + t_stim + t_delay < input_length:
            inter_pulse_period = np.random.randint(5, 25) + t_stim + 1 
            if inter_pulse_period + pulses[-1] > input_length:
                break

            channel = np.random.randint(2)
            sign = np.random.randint(2)
            inputs[i, pulses[-1], channel] = (-1)**sign*input_amp

            outputs[i, pulses[-1]:inter_pulse_period+pulses[-1],channel] = (-1)**sign*target_amp
            outputs[i, pulses[-1]:t_stim+t_delay+pulses[-1],:] = np.nan
            pulses.append(inter_pulse_period+pulses[-1])

        outputs[i, pulses[-1]:,channel] = (-1)**sign*target_amp
    mask = np.ones((N_batch, input_length, 2))
    mask[np.isnan(outputs)] = 0
    return inputs, outputs, mask, 0

def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def create_angularintegration_trials(N_batch, T, dt):
    """
    Creates N_batch trials of the angular integration task.
    Inputs are left and right angular velocity and 
    target output is sine and cosine of integrated angular velocity.
    Returns inputs, outputs, mask, 0
    -------
    """
    input_length = int(T/dt)
    X = np.expand_dims(np.linspace(-10, 10, input_length), 1)
    sigma = exponentiated_quadratic(X, X)  
    inputs = np.random.multivariate_normal(mean=np.zeros(input_length), cov=sigma, size=N_batch)
    outputs_1d =  np.cumsum(inputs, axis=1)*dt
    outputs = np.stack((np.cos(2*np.pi*outputs_1d), np.sin(2*np.pi*outputs_1d)), axis=-1)
    mask = np.ones((N_batch, input_length, 2))
    trial_params = 0
    return inputs, outputs, mask, trial_params


def create_copy_memory_trials_onehot(N_batch, N_symbols, input_length, delay):
    """    Creates N_batch trials of the copy memory task.
    N_symbols: number of symbols (K)
    input_length: number of random symbols 
    delay: timesteps before output needs to be generated
    
    Returns:
    -inputs
    -outputs
    """
    
    N_in = N_symbols+2
    N_out = N_symbols
    N_steps = delay+2*input_length
    inputs = np.zeros((N_batch, N_steps, N_in))
    outputs = np.zeros((N_batch, N_steps, N_out))

    onehot_matrix = np.eye(N_in)
    onehot_input_symbols = onehot_matrix[:-2]
    input_sequence = onehot_input_symbols[np.random.choice(N_symbols, size=(N_batch, input_length))]
    inputs[:,:input_length,:] = input_sequence                        #input sequence
    inputs[:,input_length:-1-input_length,:] = onehot_matrix[-2,:]    #Delay
    inputs[:,-1-input_length,:] = onehot_matrix[-1,:]                 #CUE

    outputs[:,-input_length:,:] = input_sequence[:,:,:-2]

    mask = np.ones((N_batch, N_steps, N_out))
    trial_params = 0
    return inputs, outputs, mask, trial_params


def create_copy_memory_trials_onehot_all(N_symbols, input_length, delay):
    """    
        Creates all possible trials of the copy memory task with N_symbols and an input sequence length of input_length.
    
    N_symbols: number of symbols (K)
    input_length: number of random symbols 
    delay: timesteps before output needs to be generated
    
    Returns:
    -inputs
    -outputs
    """
    N_batch = N_symbols**input_length
    N_in = N_symbols+2
    N_out = N_symbols
    N_steps = delay+2*input_length
    inputs = np.zeros((N_batch, N_steps, N_in))
    outputs = np.zeros((N_batch, N_steps, N_out))

    seqs = np.array(list(product(range(N_in), repeat=input_length)))
    
    onehot_matrix = np.eye(N_in)
    onehot_input_symbols = onehot_matrix[:-2]
    onehot_input_symbols = np.array([onehot_matrix[:][seqs[i,...]] for i in range(seqs.shape[0])])
    inputs[:,:input_length,:] = input_sequence                        #input sequence
    inputs[:,input_length:-1-input_length,:] = onehot_matrix[-2,:]    #Delay
    inputs[:,-1-input_length,:] = onehot_matrix[-1,:]                 #CUE

    outputs[:,-input_length:,:] = input_sequence[:,:,:-2]

    mask = np.ones((N_batch, N_steps, N_out))
    trial_params = 0
    return inputs, outputs, mask, trial_params


def create_copy_memory_trials(N_batch, N_symbols, input_length, delay):
    """
    Creates N_batch trials of the copy memory task.
    N_symbols: number of symbols (K)
    input_length: number of random symbols 
    delay: timesteps before output needs to be generated
    
    Returns:
    -inputs
    -outputs
    """
    N_out = 1
    N_steps = delay+2*input_length
    inputs = np.zeros((N_batch, delay+2*input_length))
    input_sequence = np.random.randint(1, N_symbols+1, (N_batch, input_length))
    inputs[:N_symbols] = input_sequence
    outputs[-N_symbols:] = input_sequence
    mask = np.ones((N_batch, N_steps, N_out))
    trial_params = 0
    return inputs, outputs, mask, trial_params


#Task and PerceptualDiscrimination are adapted from PsychRNN: https://github.com/murraylab/PsychRNN
class Task(ABC):
    """ The base task class.

    The base task class provides the structure that users can use to\
    define a new task. This structure is used by example tasks \
    :class:`~psychrnn.tasks.perceptual_discrimination.PerceptualDiscrimination`, \
    :class:`~psychrnn.tasks.match_to_category.MatchToCategory`, \
    and :class:`~psychrnn.tasks.delayed_discrim.DelayedDiscrimination`.

    Note:
        The base task class is not itself a functioning task. 
        The generate_trial_params and trial_function must be defined to define a new, functioning, task.

    Args:
        N_in (int): The number of network inputs.
        N_out (int): The number of network outputs.
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.

    Inferred Parameters:
        * **alpha** (*float*) -- The number of unit time constants per simulation timestep.
        * **N_steps** (*int*): The number of simulation timesteps in a trial. 

    """
    def __init__(self, N_in, N_out, dt, tau, T, N_batch):

        # ----------------------------------
        # Initialize required parameters
        # ----------------------------------
        self.N_batch = N_batch
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt
        self.tau = tau
        self.T = T

        # ----------------------------------
        # Calculate implied parameters
        # ----------------------------------
        self.alpha = (1.0 * self.dt) / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))

    def get_task_params(self):
        """ Get dictionary of task parameters.

        Note:
            N_in, N_out, N_batch, dt, tau and N_steps must all be passed to the network model as parameters -- this function is the recommended way to begin building the network_params that will be passed into the RNN model.


        Returns:
            dict: Dictionary of :class:`Task` attributes including the following keys:

            :Dictionary Keys: 
                * **N_batch** (*int*) -- The number of trials per training update.
                * **N_in** (*int*) -- The number of network inputs.
                * **N_out** (*int*) -- The number of network outputs.
                * **dt** (*float*) -- The simulation timestep.
                * **tau** (*float*) -- The unit time constant.
                * **T** (*float*) -- The trial length.
                * **alpha** (*float*) -- The number of unit time constants per simulation timestep.
                * **N_steps** (*int*): The number of simulation timesteps in a trial. 

            Note:
                The dictionary will also include any other attributes defined in your task definition.
        
        """
        return self.__dict__
    
    @abstractmethod
    def generate_trial_params(self, batch, trial):
        """ Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Args:
            batch (int): The batch number for this trial.
            trial (int): The trial number of the trial within the batch data:`batch`.

        Returns:
            dict: Dictionary of trial parameters.


        Warning:
            This function is abstract and must be implemented in a child Task object.

        Example:
            See :func:`PerceptualDiscrimination <psychrnn.tasks.perceptual_discrimination.PerceptualDiscrimination.generate_trial_params>`,\
            :func:`MatchToCategory <psychrnn.tasks.match_to_category.MatchToCategory.generate_trial_params>`,\
            and :func:`DelayedDiscrimination <psychrnn.tasks.delayed_discrim.DelayedDiscrimination.generate_trial_params>` for example implementations.

        """
        pass

    @abstractmethod
    def trial_function(self, time, params):
        """ Compute the trial properties at :data:`time`.

        Based on the :data:'params' compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`~psychrnn.tasks.task.Task.generate_trial_params`

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training.

        Warning:
            This function is abstract and must be implemented in a child Task object.

        Example:
            See :func:`PerceptualDiscrimination <psychrnn.tasks.perceptual_discrimination.PerceptualDiscrimination.trial_function>`,\
            :func:`MatchToCategory <psychrnn.tasks.match_to_category.MatchToCategory.trial_function>`,\
            and :func:`DelayedDiscrimination <psychrnn.tasks.delayed_discrim.DelayedDiscrimination.trial_function>` for example implementations.
        
        """
        pass

    
    def accuracy_function(self, correct_output, test_output, output_mask):
        """ Function to calculate accuracy (not loss) as it would be measured experimentally.

        Output should range from 0 to 1. This function is used by :class:`~psychrnn.backend.curriculum.Curriculum` as part of it's :func:`~psychrnn.backend.curriculum.default_metric`.

        Args:
            correct_output(ndarray(dtype=float, shape =(:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` ))): Correct batch output. ``y_data`` as returned by :func:`batch_generator`.
            test_output(ndarray(dtype=float, shape =(:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` ))): Output to compute the accuracy of. ``output`` as returned by :func:`psychrnn.backend.rnn.RNN.test`.
            output_mask(ndarray(dtype=bool, shape =(:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out`))): Mask. ``mask`` as returned by func:`batch_generator`.

        Returns:
            float: 0 <= accuracy <=1

        Warning:
            This function is abstract and may optionally be implemented in a child Task object.

        Example:
            See :func:`PerceptualDiscrimination <psychrnn.tasks.perceptual_discrimination.PerceptualDiscrimination.accuracy_function>`,\
            :func:`MatchToCategory <psychrnn.tasks.match_to_category.MatchToCategory.accuracy_function>`,\
            and :func:`DelayedDiscrimination <psychrnn.tasks.delayed_discrim.DelayedDiscrimination.accuracy_function>` for example implementations.
        """
        pass

    def generate_trial(self, params):
        """ Loop to generate a single trial.

        Args:
            params(dict): Dictionary of trial parameters generated by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_in` *))*) -- Trial input given :data:`params`.
            * **y_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- Correct trial output given :data:`params`.
            * **mask_trial** (*ndarray(dtype=bool, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- True during steps where the network should train to match :data:`y`, False where the network should ignore :data:`y` during training.
        """

        # ----------------------------------
        # Loop to generate a single trial
        # ----------------------------------
        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask = np.zeros([self.N_steps, self.N_out])

        for t in range(self.N_steps):
            x_data[t, :], y_data[t, :], mask[t, :] = self.trial_function(t * self.dt, params)

        return x_data, y_data, mask

    def batch_generator(self):
        """ Generates a batch of trials.

        Returns:
            Generator[tuple, None, None]:

        Yields:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.
        
        """

        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask = []
            params = []
            # ----------------------------------
            # Loop over trials in batch
            # ----------------------------------
            for trial in range(self.N_batch):
                # ---------------------------------------
                # Generate each trial based on its params
                # ---------------------------------------
                p = self.generate_trial_params(batch, trial)
                x,y,m = self.generate_trial(p)
                x_data.append(x)
                y_data.append(y)
                mask.append(m)
                params.append(p)

            batch += 1

            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(params)

    def get_trial_batch(self):
        """Get a batch of trials.

        Wrapper for :code:`next(self.batch_generator())`.

        Returns:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_in` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.

        """
        return next(self.batch_generator())


    
class PerceptualDiscrimination(Task):
    """Two alternative forced choice (2AFC) binary discrimination task. 

    On each trial the network receives two simultaneous noisy inputs into each of two input channels. The network must determine which channel has the higher mean input and respond by driving the corresponding output unit to 1.

    Takes two channels of noisy input (:attr:`N_in` = 2).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2) towards the higher mean channel.

    Loosely based on `Britten, Kenneth H., et al. "The analysis of visual motion: a comparison of neuronal and psychophysical performance." Journal of Neuroscience 12.12 (1992): 4745-4765 <https://www.jneurosci.org/content/12/12/4745>`

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        coherence (float, optional): Amount by which the means of the two channels will differ. By default None.
        direction (int, optional): Either 0 or 1, indicates which input channel will have higher mean input. By default None.

    """

    def __init__(self, N_batch, training_kwargs, low=0.0, high=1.):
        super(PerceptualDiscrimination,self).__init__(2, 2, 10, 100, 2000, N_batch)
        
        self.coherence = training_kwargs['coherence']
        if np.any(training_kwargs['coherence_list'] == None):
            self.coherence_list = [0.1, 0.3, 0.5, 0.7]
        else:
            self.coherence_list = training_kwargs['coherence_list']
            
        if training_kwargs['onset_times'] == None:
            self.onset_times = None
        else:
            self.onset_times = training_kwargs['onset_times']
            
        if training_kwargs['stim_durations'] == None:
            self.stim_durations = None
        else:
            self.stim_durations = training_kwargs['stim_durations']

        # self.direction = direction
        self.lo = low # Low value for one hot encoding
        self.hi = high # High value for one hot encoding

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **coherence** (*float*) -- Amount by which the means of the two channels will differ. :attr:`self.coherence` if not None, otherwise ``np.random.exponential(scale=1/5)``.
                * **direction** (*int*) -- Either 0 or 1, indicates which input channel will have higher mean input. :attr:`self.direction` if not None, otherwise ``np.random.choice([0, 1])``.
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .1.
                * **onset_time** (*float*) -- Stimulus onset time. ``np.random.random() * self.T / 2.0``.
                * **stim_duration** (*float*) -- Stimulus duration. ``np.random.random() * self.T / 4.0 + self.T / 8.0``.

        """

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        if self.coherence == None:
            params['coherence'] = np.random.choice(self.coherence_list)
        else:
            params['coherence'] = self.coherence
        params['direction'] = np.random.choice([0, 1])
        params['stim_noise'] = 0.1
        
        if self.onset_times == None:
            params['onset_time'] = np.random.random() * self.T / 2.0
        else:
             params['onset_time'] = np.random.choice(self.onset_times)
                
        if self.stim_durations == None:
            params['stim_duration'] = np.random.random() * self.T / 4.0 + self.T / 8.0
        else:
            params['stim_duration']  = np.random.choice(self.stim_durations)
            

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. For ``params['onset_time'] < time < params['onset_time'] + params['stim_duration']`` , 1 is added to the noise in both channels, and :data:`params['coherence']` is also added in the channel corresponding to :data:`params[dir]`.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. From ``time > params['onset_time'] + params[stim_duration] + 20`` onwards, the correct output is encoded using one-hot encoding. Until then, y_t is 0 in both channels.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for ``time > params['onset_time'] + params['stim_duration']`` and False otherwise.

        """

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        coh = params['coherence']
        onset = params['onset_time']
        stim_dur = params['stim_duration']
        dir = params['direction']
        
        
        # x_t[onset:onset + stim_dur] += 1 + coh

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if onset < t < onset + stim_dur:
            x_t[dir] += 1 + coh
            x_t[(dir + 1) % 2] += 1

        if t > onset + stim_dur + 20:
            y_t[dir] = self.hi
            y_t[1-dir] = self.lo

        if t < onset + stim_dur:
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t
    
    

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))


    
    
    
#
class PoissonClicks(Task):
    """Possion clicks task

    On each trial the network receives two simultaneous noisy inputs into each of two input channels. The network must determine which channel has the higher mean input and respond by driving the corresponding output unit to 1.

    Takes two channels of noisy input (:attr:`N_in` = 2).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2) towards the higher mean channel.

    Based on Brunton, Bingni W., Matthew M. Botvinick, and Carlos D. Brody. "Rats and humans can optimally accumulate evidence for decision-making." 
                        Science 340.6128 (2013): 95-98.
                        
    The sum of the two pulse rates was kept fixed within each task, and discrimination difficulty was controlled on each trial by the ratio of the two rates

    longest stimulus duration used [1 s for rats, 4s for humans].
    
    we computed the excess pulse rate difference (right pulses/s – left pulses/s, relative to the value expected given the random processes used to generate the trial) and then obtained an average for trials resulting in a right (red) and an average for trials resulting in a left (green) decision
    
    Args:
        dt (float): The simulation timestep.
        T (float): The trial length in ms.
        N_batch (int): The number of trials for dataset.
        sum_of_rates: sum of the two pulse rates, fixed within each task (i.e. experiment).
        ratio: ratio of the two rates. Determines two pulse rates based on sum_of_rates. If a list is given, this is used for all trials. 
        lambdas: the mean activation of the different channels in Hz.
        exp_trunc_params is a dictionary with keys: b, scale, loc
    """
    def __init__(self, N_batch, training_kwargs, low=0., hi=1.):
        super(PoissonClicks,self).__init__(2, 2, 10, 100, 2000, N_batch)
        if training_kwargs['model_type'] == 'rnn_cued':
            self.cued = True
        else:
            self.cued = False
        self.N_batch = N_batch
        self.dt = training_kwargs['dt']
        self.T = training_kwargs['T']
        self.low = low
        self.hi  = hi
        # self.fixed_startcue_onsetandduration = training_kwargs['fixed_startcue_onsetandduration']
        self.fixed_cue_onsetandduration = training_kwargs['fixed_cue_onsetandduration']
        self.fixed_stim_duration = training_kwargs['fixed_stim_duration']
        self.fixed_stim_duration_list = training_kwargs['fixed_stim_duration_list']
        self.zero_target = training_kwargs['zero_target']
        
        self.ratio = training_kwargs['ratio'] #range = (-\infty,\infty)
        self.ratios = training_kwargs['ratios']
        self.sum_of_rates = training_kwargs['sum_of_rates']
        if training_kwargs['exp_trunc_params'] == {}:
            self.exp_trunc_params = {'b':1., 'scale':1., 'loc':0.}
        else:
            self.exp_trunc_params = training_kwargs['exp_trunc_params'] 
        self.N_steps = int(np.ceil(self.T / self.dt))
        self.clicks_capped = training_kwargs['clicks_capped']
        self.training_kwargs = training_kwargs
    
    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **lambdas** (*float*) -- Amount by which the means of the two channels will differ. :attr:`self.coherence` if not None, otherwise ``np.random.exponential(scale=1/5)``.
                * **onset_time** (*float*) -- Stimulus onset time. ``np.random.random() * self.T / 10.``.
                * **stim_duration** (*float*) -- Stimulus duration. ``Truncated exponential with parameters b, loc, scale``.
        
        NOTE: if self.fixed_cue_duration>=self.N_steps, then cue_duration is until end of trial

        """
        params = dict()
        
        if self.ratio == None:
            params['ratio'] = np.random.choice(self.ratios)
        elif self.ratio != None:
            params['ratio'] = self.ratio
        else:
            print("No ratio or ratios defined.")
            
            
        
        if self.fixed_stim_duration_list != None:# full list of all timings?:yes
            params['stim_durations_and_pauses'] = self.fixed_stim_duration_list
            full_stim_duration = np.sum(params['stim_durations_and_pauses'])
            if self.fixed_cue_onsetandduration != None:
                params['onset_cue'] = self.fixed_cue_onsetandduration[0]
                params['cue_duration'] = self.fixed_cue_onsetandduration[1]
                # params['stim_duration'] = self.fixed_stim_duration_list[0]
                params['stim_durations_and_pauses'] = [params['onset_cue']+params['cue_duration']]
                params['stim_durations_and_pauses'].extend(self.fixed_stim_duration_list)
                
        else:
            if self.fixed_cue_onsetandduration != None:# list of fixed cue timing and duration? 
                params['onset_cue'] = self.fixed_cue_onsetandduration[0]
                params['cue_duration'] = self.fixed_cue_onsetandduration[1]

            else:
                params['onset_cue'] =  int(np.random.random() * self.T / 20. /self.dt) #start in first tenth of the trial
                params['cue_duration'] = max(1,int(np.random.random() * self.T / 20. /self.dt)) #duration is tenth of the trial
            
        
            if self.fixed_stim_duration != None: # fixed stimulus duration?yes
                params['stim_duration'] = self.fixed_stim_duration
            else:                                # fixed stimulus duration?no
                params['stim_duration'] = int(truncexpon.rvs(b=self.exp_trunc_params['b'],
                                                         scale=self.exp_trunc_params['scale'],
                                                         loc=self.exp_trunc_params['loc'],
                                                         size=1)/self.dt)
                
            params['stim_durations_and_pauses'] = [params['onset_cue']+params['cue_duration'], params['stim_duration']]
            
        params['stim_duration'] = np.sum(np.array(params['stim_durations_and_pauses'][1:]))
        
        if self.fixed_cue_onsetandduration == None:
            if self.fixed_stim_duration_list != None:
                params['output_cue'] = params['onset_cue'] + params['cue_duration'] + params['stim_duration'] + self.fixed_stim_duration_list[-2]
                params['output_duration'] = self.fixed_stim_duration_list[-1]
            else:
                params['output_cue'] = params['onset_cue'] + params['cue_duration'] + params['stim_duration'] + max(1,int(np.random.random() * self.T / 5. /self.dt))
                params['output_duration'] =  max(1,int(np.random.random() * self.T / 5. /self.dt))
        else:
            params['output_cue'] = params['onset_cue'] + params['cue_duration'] + params['stim_duration'] + self.fixed_cue_onsetandduration[2]
            params['output_duration'] = self.fixed_cue_onsetandduration[3]
        
        

        #calculate rates for given ration and sum
        lambdas = [self.sum_of_rates/(1+1/abs(params['ratio']))]
        lambdas.insert(int(np.sign(params['ratio'])), self.sum_of_rates - lambdas[0])
        params['lambdas'] = np.array(lambdas) 

        return params
    
    

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. For ``params['onset_time'] < time < params['onset_time'] + params['stim_duration']`` , 1 is added to the noise in both channels, and :data:`params['coherence']` is also added in the channel corresponding to :data:`params[dir]`.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. From ``time > params['onset_time'] + params[stim_duration] + 20`` onwards, the correct output is encoded using one-hot encoding. Until then, y_t is 0 in both channels.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for ``time > params['onset_time'] + params['stim_duration']`` and False otherwise.
        """

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.zeros(self.N_out)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        ratio = params['ratio']
        lambdas = params['lambdas']
        onset = params['onset_cue']
        stim_dur = params['stim_duration']
        # dir = params['direction']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if onset < t < onset + stim_dur:
            if np.random.rand() < lambdas[0]:
                x_t[dir] += 1 
            if np.random.rand() < lambdas[1]:
                x_t[(dir + 1) % 2] += 1 

        if t > onset + stim_dur + 20:
            y_t[dir] = self.hi
            y_t[1-dir] = self.lo

        if t < onset + stim_dur:
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t

    
    def generate_trial(self, params):
        """ Loop to generate a single trial.

        Args:
            params(dict): Dictionary of trial parameters generated by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_in` *))*) -- Trial input given :data:`params`.
            * **y_trial** (*ndarray(dtype=float, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- Correct trial output given :data:`params`.
            * **mask_trial** (*ndarray(dtype=bool, shape=(*:attr:`N_steps`, :attr:`N_out` *))*) -- True during steps where the network should train to match :data:`y`, False where the network should ignore :data:`y` during training.
        """
        lambdas = params['lambdas']
        
        all_input =  np.zeros([self.N_steps, 4])
        x_data = np.zeros([self.N_steps, 2])
        if self.zero_target == 0:
            y_data = np.zeros([self.N_steps, self.N_out])
        else:
            y_data = np.ones([self.N_steps, self.N_out])/2.
        mask = np.zeros([self.N_steps, self.N_out])

        #make sure this is higher than zero
        stim_cue_start = params['onset_cue'] - params['cue_duration']
        if stim_cue_start<0:
            stim_cue_start=0
        stim_cue_duration = params['cue_duration']
        all_input[:, 2][stim_cue_start:stim_cue_start+stim_cue_duration] = 1.
        
        elapsed_time = 0
        x_stims = []
        for i in range(0,len(params['stim_durations_and_pauses']), 2):
            elapsed_time += params['stim_durations_and_pauses'][i]
            stim_duration = params['stim_durations_and_pauses'][i+1]
            x_data[elapsed_time:elapsed_time+stim_duration] = np.random.poisson(lam=lambdas*self.dt/1000, size=(stim_duration,2))
            elapsed_time += stim_duration
            
        if self.clicks_capped == True:
            x_data = np.where(x_data<2, x_data, 1)

            
        #ticks per channel
        N_clicks = np.sum(x_data, axis=0)
        #determine N_1<N_2
        if N_clicks[0] == N_clicks[1]:
            highest_click_count_index = np.random.choice([0,1]) #if N_1=N_2 choose reward randomly
            if self.training_kwargs['equal_clicks'] == 'addone_random':
                time_point = stim_duration+1 #np.random.random_integers(low=0, high=stim_duration)
                x_data[elapsed_time+1, highest_click_count_index] = 1.
                N_clicks[highest_click_count_index] += 1
        else:
            highest_click_count_index = np.argmax(N_clicks)
            
        all_input[:, :2] = x_data
            
        all_input[:, 3][params['output_cue']-params['cue_duration']:params['output_cue']] = 1.
        y_data[params['output_cue']:params['output_cue']+params['output_duration'], highest_click_count_index] = 1.
        y_data[params['output_cue']:params['output_cue']+params['output_duration'], 1-highest_click_count_index] = 0.

        if self.training_kwargs['count_out'] == "1D_uncued":
            y_data = np.cumsum(x_data, 0)[:,1]-np.cumsum(x_data, 0)[:,0]
            if self.training_kwargs['input_noise']:
                x_data += np.random.uniform(-self.training_kwargs['input_noise'], self.training_kwargs['input_noise'], x_data.shape)
            if self.training_kwargs['input_noise_startpulse_steps']:
                x_data[:self.training_kwargs['input_noise_startpulse_steps'],:] += np.random.uniform(-self.training_kwargs['input_noise'], self.training_kwargs['input_noise'], (self.training_kwargs['input_noise_startpulse_steps'], 2))
            mask[:] = 1
            return x_data, y_data, mask

        elif self.training_kwargs['count_out'] == "2D_uncued":
            y_data = np.cumsum(x_data, 0)       
            if self.training_kwargs['input_noise']:
                x_data += np.random.uniform(-self.training_kwargs['input_noise'], self.training_kwargs['input_noise'], x_data.shape)
            mask[:] = 1
            return x_data, y_data, mask
            
        if self.training_kwargs['target_withcue'] == True:
            y_data[params['output_cue']-params['cue_duration']:params['output_cue'], highest_click_count_index] = 1.
            mask[params['output_cue']-params['cue_duration']:params['output_cue'], :] = 1.
            
        if self.training_kwargs['equal_clicks'] == 'mask':
            0 #TODO
        elif self.training_kwargs['equal_clicks'] == 'equal_output' and N_clicks[0] == N_clicks[1]:
            y_data[params['output_cue']:params['output_cue']+params['output_duration'], :] = 0.5
            if self.training_kwargs['target_withcue'] == True:
                y_data[params['output_cue']-params['cue_duration']:params['output_cue'], :] = 0.5
        else:
            mask[params['output_cue']:params['output_cue']+params['output_duration'],:] = 1.
        
        if not self.cued:
            return x_data, y_data, mask
        else:
            return all_input, y_data, mask

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))


  

#
# class PoissonClicks2():
#     def __init__(self, N_batch, training_kwargs):
#         self.N_batch = N_batch
#         self.dt = training_kwargs['dt']
#         self.T = training_kwargs['T']
#         self.N_steps = int(np.ceil(self.T / self.dt))
#         self.ratio = training_kwargs['ratio'] #ratios
#         self.sum_of_rates = training_kwargs['sum_of_rates']
#         self.clicks_capped = training_kwargs['clicks_capped']
        
#         self.min_stim_start =  int(training_kwargs['T'] / 20. /self.dt)
#         self.max_stim_start =  int(training_kwargs['T'] / 10. /self.dt)
#         self.min_stim = int(training_kwargs['min_stim'] / self.dt)
#         self.max_stim = int(training_kwargs['max_stim'] / self.dt)
#         self.cue_duration = int(training_kwargs['cue_duration']/ self.dt)
#         self.target_duration = int(training_kwargs['target_duration']/ self.dt)

#         self.lambdas = [self.sum_of_rates/(1+1/abs(training_kwargs['ratio']))]
#         self.lambdas.insert(int(np.sign(training_kwargs['ratio'])), self.sum_of_rates - self.lambdas[0])
#         self.lambdas = np.array(self.lambdas)
        
#         self.fixed_cue_onsetandduration = training_kwargs['fixed_cue_onsetandduration']
#         self.fixed_stim_duration = training_kwargs['fixed_stim_duration']
#         self.fixed_stim_duration_list = training_kwargs['fixed_stim_duration_list']
#         self.zero_target = training_kwargs['zero_target']
        
#         self.ratio = training_kwargs['ratio'] 
#         self.ratios = training_kwargs['ratios']
#         self.sum_of_rates = training_kwargs['sum_of_rates']
#         if training_kwargs['exp_trunc_params'] == {}:
#             self.exp_trunc_params = {'b':1., 'scale':1., 'loc':0.}
#         else:
#             self.exp_trunc_params = training_kwargs['exp_trunc_params'] 
    
#     def get_trial_batch(self):
#         trial_params = []
#         all_inputs = np.zeros((self.N_batch, self.N_steps, 3))
#         y = np.ones((self.N_batch,self.N_steps, 2))/2.
#         mask = np.zeros((self.N_batch,self.N_steps, 2))

#         for i in range(self.N_batch):
#             params = {}
#             stim_start =    int(np.random.random() * (self.max_stim_start - self.min_stim_start)) + self.cue_duration + self.min_stim_start
#             stim_duration = int(np.random.random() * (self.max_stim-self.min_stim)) + self.min_stim

#             x_data = np.zeros([self.N_steps, 2])
#             x_data[stim_start:stim_start+stim_duration] = np.random.poisson(lam=self.lambdas*self.dt/1000, size=(stim_duration,2))
            
#             if self.clicks_capped == True:
#                 x_data = np.where(x_data<2, x_data, 1)

#             #flip 50%
#             params["first_channel"] = 0
#             if np.random.random()<0.5:
#                 params["first_channel"] = 1
#                 x_data[:, [0,1]] = x_data[:, [1,0]]


#             params['stim_start'] = stim_start
#             params['stim_duration'] = stim_duration
#             params['switch_times'] = switch_times
#             params['n_switches'] = n_switches
#             params['target_state'] = (params["first_channel"] + n_switches) % 2
#             trial_params.append(params.copy())

#             all_inputs[i, :, :2] = x_data
#             all_inputs[i, stim_start-self.cue_duration:stim_start+stim_duration, -1] = 1

#             y[i, stim_start+stim_duration:stim_start+stim_duration+self.target_duration, params['target_state']] = 1.
#             y[i, stim_start+stim_duration:stim_start+stim_duration+self.target_duration, 1-params['target_state']] = 0.
            
#             mask[i, stim_start+stim_duration:stim_start+stim_duration+self.target_duration, :] = 1
#         return all_inputs, y, mask, np.array(trial_params)


class DynamicPoissonClicks():
    def __init__(self, N_batch, training_kwargs):
        self.N_batch = N_batch
        self.dt = training_kwargs['dt']
        self.T = training_kwargs['T']
        self.N_steps = int(np.ceil(self.T / self.dt))
        self.ratio = training_kwargs['ratio'] #ratios
        self.sum_of_rates = training_kwargs['sum_of_rates']
        self.hazard_rate =  training_kwargs['hazard_rate']
        self.clicks_capped = training_kwargs['clicks_capped']
        
        self.min_stim_start =  int(training_kwargs['T'] / 20. / self.dt)
        self.max_stim_start =  int(training_kwargs['T'] / 10. / self.dt)
        self.min_stim = int(training_kwargs['min_stim'] / self.dt)
        self.max_stim = int(training_kwargs['max_stim'] / self.dt)
        self.cue_duration = int(training_kwargs['cue_duration']/ self.dt)
        self.target_duration = int(training_kwargs['target_duration']/ self.dt)

        # self.lambdas = training_kwargs['lambdas']  # lambdas = [38, 2]
        self.lambdas = [self.sum_of_rates/(1+1/abs(training_kwargs['ratio']))]
        self.lambdas.insert(int(np.sign(training_kwargs['ratio'])), self.sum_of_rates - self.lambdas[0])
        self.lambdas = np.array(self.lambdas)
    
    def get_trial_batch(self):
        trial_params = []
        all_inputs = np.zeros((self.N_batch, self.N_steps, 3))
        y = np.ones((self.N_batch,self.N_steps, 2))/2.
        mask = np.zeros((self.N_batch,self.N_steps, 2))

        for i in range(self.N_batch):
            params = {}
            stim_start =    int(np.random.random() * (self.max_stim_start - self.min_stim_start)) + self.cue_duration + self.min_stim_start
            stim_duration = int(np.random.random() * (self.max_stim-self.min_stim)) + self.min_stim

            x_data = np.zeros([self.N_steps, 2])
            x_data[stim_start:stim_start+stim_duration] = np.random.poisson(lam=self.lambdas*self.dt/1000, size=(stim_duration,2))
            
            if self.clicks_capped == True:
                x_data = np.where(x_data<2, x_data, 1)

            switches = np.random.random((stim_duration)) < self.dt/1000.*self.hazard_rate
            switch_times = np.where(switches>0)[0]
            n_switches = len(switch_times)

            #flip 50/50
            params["first_channel"] = 0
            if np.random.random()<0.5:
                params["first_channel"] = 1
                x_data[:, [0,1]] = x_data[:, [1,0]]

            for switch in switch_times:
                x_data[switch:, [0,1]] = x_data[switch:, [1,0]]

            params['stim_start'] = stim_start
            params['stim_duration'] = stim_duration
            params['switch_times'] = switch_times
            params['n_switches'] = n_switches
            params['target_state'] = (params["first_channel"] + n_switches) % 2
            trial_params.append(params.copy())

            all_inputs[i, :, :2] = x_data
            all_inputs[i, stim_start-self.cue_duration:stim_start+stim_duration, -1] = 1

            y[i, stim_start+stim_duration:stim_start+stim_duration+self.target_duration, params['target_state']] = 1.
            y[i, stim_start+stim_duration:stim_start+stim_duration+self.target_duration, 1-params['target_state']] = 0.
            
            mask[i, stim_start+stim_duration:stim_start+stim_duration+self.target_duration, :] = 1
        return all_inputs, y, mask, np.array(trial_params)
    
    
# class ReadySetGo():
#     """
#     Read-set-go task with random hold-delay.
    
#     """
#     def __init__(self, **training_kwargs):
#         self.pulse_duration = 
        
    
    # def get_trial_batch(self):

#     return all_inputs, y, mask, trial_params






