import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import Math, HTML, display, Latex, YouTubeVideo, clear_output
import matplotlib
import numpy as np
import time
from os import path


def embed_video(url, labels={}):

    names = list(labels.keys())
    buttons = []

    def button_clicked(change):
        clear_output()
        display(YouTubeVideo('ARJ8cAGm6JE', start=(
            labels[change.description]), autoplay=True))
        display(button_box)

    for i in range(len(names)):
        buttons.append(widgets.Button(description=names[i]))
        buttons[-1].on_click(button_clicked)

    button_box = widgets.HBox(buttons)

    display(YouTubeVideo('ARJ8cAGm6JE', autoplay=False))
    display(button_box)


def toggle_code(title="code", on_load_hide=True, above=1):
    if above < 1:
        print("Error: please input a valid value for 'above'")
    else:
        display_string = """
    <script>
      function get_new_label(butn, hide) {
          var shown = $(butn).parents("div.cell.code_cell").find('div.input').is(':visible');
          var title = $(butn).val().substr($(butn).val().indexOf(" ") + 1)
          return ((shown) ? 'Show ' : 'Hide ') + title
      }
      function code_toggle(butn, hide) {
        $(butn).val(get_new_label(butn,hide));
        $(hide).slideToggle();
      };
    </script>
    <input type="submit" value='initiated' class='toggle_button'>
    <script>
        var hide_area = $(".toggle_button[value='initiated']").parents('div.cell').prevAll().addBack().slice(-""" + str(above) + """)
        hide_area = $(hide_area).find("div.input").add($(hide_area).filter("div.text_cell"))
        $(".toggle_button[value='initiated']").prop("hide_area", hide_area)
        $(".toggle_button[value='initiated']").click(function(){
            code_toggle(this, $(this).prop("hide_area"))
        }); 
$(".toggle_button[value='initiated']").parents("div.output_area").insertBefore($(".toggle_button[value='initiated']").parents("div.output").find('div.output_area').first());
    var shown = $(".toggle_button[value='initiated']").parents("div.cell.code_cell").find('div.input').is(':visible');
    var title = ((shown) ? 'Hide ' : 'Show ') + '""" + title + """'; 
    """
    if on_load_hide:
        display_string += """ $(".toggle_button[value='initiated']").addClass("init_show");
            $(hide_area).addClass("init_hidden"); """
    else:
        display_string += """ $(".toggle_button[value='initiated']").addClass("init_hide");
            $(hide_area).addClass("init_shown"); """

    display_string += """ $(".toggle_button[value='initiated']").val(title);
    </script>"""
    display(HTML(display_string))


def dropdown_math(title, text=None, file=None):
    out = widgets.Output()
    with out:
        if not(file == None):
            handle = open(file, 'r')
            display(Latex(handle.read()))
        else:
            display(Math(text))
    accordion = widgets.Accordion(children=[out])
    accordion.set_title(0, title)
    accordion.selected_index = None
    return accordion


def remove_axes(which_axes=''):

    frame = plt.gca()
    if 'x' in which_axes:
        frame.axes.get_xaxis().set_visible(False)
    if 'y' in which_axes:
        frame.axes.get_yaxis().set_visible(False)

    elif which_axes == '':
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

    return


def set_notebook_preferences(home_button = True):

    css_file = path.join(path.dirname(__file__), 'notebook.css')
    css = open(css_file, "r").read()

    display_string = "<style>" + css + "</style>"

    if (home_button):
        display_string += """
     <input type="submit" value='Home' id="initiated" class='home_button' onclick='window.location="../index.ipynb"' style='float: right; margin-right: 40px;'>
    <script>
    $('.home_button').not('#initiated').remove();
    $('.home_button').removeAttr('id');
    $(".home_button").insertBefore($("div.cell").first());
    """

    else: 
        display_string += "<script>$('.home_button').remove();"

    display_string += """
    $('div.input.init_hidden').hide()
    $('div.input.init_shown').show()
    $('.toggle_button').each(function( index, element ) {
       var prefix;
       if (this.classList.contains('init_show')) {
           prefix = 'Show '
       }
       else if (this.classList.contains('init_hide')) {
           prefix = 'Hide '
       };
       $(this).val(prefix + $(this).val().substr($(this).val().indexOf(" ") + 1))
    });
    IPython.OutputArea.prototype._should_scroll = function(lines) {
        return false;
    }
    </script>
    """
    display(HTML(display_string))

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'sans-serif'

    matplotlib.rc('axes', titlesize=14)
    matplotlib.rc('axes', labelsize=14)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)


def beautify_plot(params):

    if not(params.get('title', None) == None):
        plt.title(params.get('title'))

    if not(params.get('x', None) == None):
        plt.xlabel(params.get('x'))

    if not(params.get('y', None) == None):
        plt.ylabel(params.get('y'))


def sample_weights_from(w1, w2, post):
    idx1, idx2 = np.arange(0, post.shape[0]), np.arange(0, post.shape[1])
    idx1, idx2 = np.meshgrid(idx1, idx2)
    idx = np.stack([idx1, idx2], axis=2)
    idx = np.reshape(idx, (-1, 2))

    flat_post = np.reshape(post, (-1,)).copy()
    flat_post /= flat_post.sum()

    sample_idx = np.random.choice(
        np.arange(0, flat_post.shape[0]), p=flat_post)
    grid_idx = idx[sample_idx]

    return w1[grid_idx[0]], w2[grid_idx[1]]


def kNN(X_train, Y_train, X_test, k, p=2):

    # clone test points for comparisons as before
    X_test_clone = np.stack([X_test]*X_train.shape[0], axis=-2)
    distances = np.sum(np.abs(X_test_clone - X_train)**p,
                       axis=-1)  # compute Lp distances
    idx = np.argsort(distances, axis=-1)[:, :k]  # find k smallest distances
    classes = Y_train[idx]  # classes corresponding to the k smallest distances
    predictions = []

    for class_ in classes:
        uniques, counts = np.unique(class_, return_counts=True)

        if (counts == counts.max()).sum() == 1:
            predictions.append(uniques[np.argmax(counts)])
        else:
            predictions.append(np.random.choice(
                uniques[np.where(counts == counts.max())[0]]))

    return np.array(predictions)


def sig(x):
    return 1/(1 + np.exp(-x))


def logistic_gradient_ascent(x, y, init_weights, no_steps, stepsize):
    x = np.append(np.ones(shape=(x.shape[0], 1)), x, axis=1)
    w = init_weights.copy()
    w_history, log_liks = [], []

    for n in range(no_steps):
        log_liks.append(np.sum(y*np.log(sig(x.dot(w))) +
                               (1 - y)*np.log(1 - sig(x.dot(w)))))
        w_history.append(w.copy())

        sigs = sig(x.dot(w))
        dL_dw = np.mean((y - sigs)*x.T, axis=1)
        w += stepsize*dL_dw

    return np.array(w_history), np.array(log_liks)


def softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x), axis=1)).T


def softmax_gradient_ascent(x, y, init_weights, no_steps, stepsize):
    x = np.append(np.ones(shape=(x.shape[0], 1)), x, axis=1)
    w = init_weights.copy()
    w_history, log_liks = [], []

    for n in range(no_steps):
        log_liks.append(np.sum(y*np.log(softmax(x.dot(w)))))
        w_history.append(w.copy())

        soft_ = softmax(x.dot(w))
        dL_dw = (x.T).dot(y - soft_)/x.shape[0]
        w += stepsize*dL_dw

    return np.array(w_history), np.array(log_liks)


def PCA(x):

    S = ((x - x.mean()).T).dot(x - x.mean())/x.shape[0]
    eig_values, eig_vectors = np.linalg.eig(S)
    sort_idx = (-eig_values).argsort()
    eig_values, eig_vectors = eig_values[sort_idx], eig_vectors[:, sort_idx]

    return np.real(eig_values), np.real(eig_vectors)


def PCA_N(x):

    S = ((x - x.mean(axis=0)).T).dot(x - x.mean(axis=0))/x.shape[0]

    t = time.time()
    eig_values, eig_vectors = np.linalg.eig(S)
    print('Time taken for high-dimensional approach:',
          np.round((time.time() - t), 3), 'sec')

    sort_idx = (-eig_values).argsort()
    eig_values, eig_vectors = eig_values[sort_idx], eig_vectors[:, sort_idx]

    return np.real(eig_values), np.real(eig_vectors)


def k_means(x, K, max_steps, mu_init):
    N, D = x.shape
    mu = mu_init.copy()

    s = np.zeros(shape=(N, K))
    assignments = np.random.choice(np.arange(0, K), N)
    s[np.arange(s.shape[0]), assignments] = 1

    x_stacked = np.stack([x]*K, axis=1)
    losses = [np.sum(s*np.sum((x_stacked - mu)**2, axis=2))]
    converged = False

    for i in range(max_steps):

        mus = (s.T).dot(x)
        s_sum = s.sum(axis=0).reshape((-1, 1))
        s_sum[np.where(s_sum < 1)] = 1
        mus /= s_sum

        distances = np.sum((x_stacked - mus)**2, axis=2)
        min_idx = np.argmin(distances, axis=1)
        s_prev = s.copy()
        s = np.zeros_like(s)
        s[np.arange(s.shape[0]), min_idx] = 1

        losses.append(np.sum(s*np.sum((x_stacked - mus)**2, axis=2)))

        if np.prod(np.argmax(s, axis=1) == np.argmax(s_prev, axis=1)):
            break

    return s, mus, losses
