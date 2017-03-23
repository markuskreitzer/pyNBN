# coding=utf-8



for num in range(1,21):
    figure_name = 'step3_fig{0}.png'.format(num)
    caption= 'Grid search for 20 figures.'
    label = figure_name.rstrip('.png')
    boo="""
\\begin{{figure}}[H]
\centering
\includegraphics{{img/{figure_name}}}
\caption{{{caption}}}
\label{{fig:{label}}}
\end{{figure}}
""".format(figure_name=figure_name, caption=caption, label=label)
    boo = """
    \\tcbincludegraphics{{img/step3_fig{0}.png}}""".format(num)
    print boo,
