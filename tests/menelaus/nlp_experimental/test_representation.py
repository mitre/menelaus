from menelaus.nlp_experimental.representation import ExampleNLPRepresentation


def test_nlp_rep_default_init():
    """ ensure internal pre/post-processors default to empty list """
    rep = ExampleNLPRepresentation()
    assert rep.preprocessors == []
    assert rep.postprocessors == []

def test_nlp_rep_full_pipeline():
    """ ensure all pre/post-processors apply when given """
    pre = lambda x: x*2
    post = lambda x: x*3
    rep = ExampleNLPRepresentation(preprocessors=[pre], postprocessors=[post])
    data = 1
    new_data = rep.transform(data)
    # check that preprocessor, transform fn., postprocessor all worked
    assert new_data == ((data*2)+1) * 3 