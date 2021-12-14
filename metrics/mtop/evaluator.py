from .top_metrics import top_metrics


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}

        # Here is the post-process code from huggingface transformers,
        # it clean up clean up the space from tokenization(" 's"->"'s" e.t.c)
        # which is different from tensorflow.

        # @staticmethod
        # def clean_up_tokenization(out_string: str) -> str:
        # """
        # Clean up a list of simple English tokenization artifacts like spaces
        # before punctuations and abbreviated forms.
        #
        # Args:
        #     out_string (:obj:`str`): The text to clean up.
        #
        # Returns:
        #     :obj:`str`: The cleaned-up string.
        # """
        # out_string = (
        #     out_string.replace(" .", ".")
        #     .replace(" ?", "?")
        #     .replace(" !", "!")
        #     .replace(" ,", ",")
        #     .replace(" ' ", "'")
        #     .replace(" n't", "n't")
        #     .replace(" 'm", "'m")
        #     .replace(" 's", "'s")
        #     .replace(" 've", "'ve")
        #     .replace(" 're", "'re")
        # )
        # return out_string

        # Therefore, to make fair comparison, we "convert that back".

        preds = [pred.replace("'s", " 's").replace(",", " ,").replace("?", " ?").replace("a.m.",
                                                                                         "a.m .").replace(
            "p.m.", "p.m .").replace("p. m.", "p . m .").replace("Dr.", "Dr .").replace("Mrs.", "Mrs .").replace("Mr.",
                                                                                                                 "Mr .").replace(
            "O.J.", "O.J .").replace("Jr.", "Jr .").replace("N.", "N .").replace("J.", "J .").replace("Y.",
                                                                                                      "Y .").replace(
            "drs.", "drs .").replace("St.", "St .").replace("Portugal.", "Portugal .") for pred in preds]

        golds = [gold["seq_out"] for gold in golds]
        eval_results = top_metrics(golds, preds)
        summary["exact_match"] = eval_results["full_accuracy"]
        summary["template_accuracy"] = eval_results["intent_arg_accuracy"]
        return summary
