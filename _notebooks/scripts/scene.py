import torch
import numpy as np
from manim import *


class EmbeddingExample(Scene):
    initial_text = "The quick brown fox jumps over the lazy dog!!"
    post_process = initial_text.strip(",|!").lower().strip()
    tokenized = post_process.split()

    def construct(self):
        title = TextMobject("Define some initial text")
        title.to_corner(UP + LEFT)
        first_step = TextMobject(self.initial_text)
        self.play(
            Write(title),
            FadeInFrom(first_step, DOWN),
        )
        self.wait(1)
        second_title = TextMobject("Preprocessing (optional)")
        second_title.to_corner(UP + LEFT)
        second_step = TextMobject(self.post_process, color=BLUE)
        self.play(
            Transform(title, second_title),
            ReplacementTransform(first_step, second_step),
        )
        third_title = TextMobject("Tokenize it")
        third_title.to_corner(UP + LEFT)
        first_arrow = Arrow(DOWN, 3 * DOWN, color=BLUE)
        first_arrow.next_to(second_step, DOWN)
        third_step = TextMobject(str(self.tokenized))
        third_step.next_to(first_arrow, DOWN)
        self.play(GrowArrow(first_arrow))
        self.play(
            Transform(second_title, third_title), FadeOut(title), FadeIn(third_step)
        )

        fourth_step = TextMobject(str(self.tokenized))
        self.wait(3)
        self.play(
            FadeOut(second_step),
            FadeOut(first_arrow),
            FadeOut(third_step),
            Transform(third_step, fourth_step),
        )
        fith_step_text = VGroup(
            TextMobject("the"),
            TextMobject("quick"),
            TextMobject("brown"),
            TextMobject("fox"),
            TextMobject("jumps"),
            TextMobject("over"),
            TextMobject("the"),
            TextMobject("lazy"),
            TextMobject("dog"),
        ).arrange(DOWN, aligned_edge=LEFT)
        self.play(ReplacementTransform(fourth_step, fith_step_text))
        second_arrow = Arrow(RIGHT, 3 * RIGHT)
        second_arrow.next_to(fith_step_text, RIGHT)
        fith_step_number = VGroup(
            TextMobject("0"),
            TextMobject("1"),
            TextMobject("2"),
            TextMobject("3"),
            TextMobject("4"),
            TextMobject("5"),
            TextMobject("0"),
            TextMobject("6"),
            TextMobject("7"),
        ).arrange(DOWN, aligned_edge=LEFT)
        fith_step_number.next_to(second_arrow, RIGHT)
        fourth_title = TextMobject("Map each word to an Integer*")
        second_line = TextMobject("*Notice that two words")
        third_line = TextMobject("the", color=RED)
        fourth_line = TextMobject(" were mapped to 0")
        second_line.scale(0.6)
        third_line.scale(0.6)
        fourth_line.scale(0.6)
        fourth_title.to_corner(UP + LEFT)
        # Position text
        second_line.next_to(fourth_title, DOWN)
        second_line.to_edge(LEFT)
        third_line.next_to(second_line, 0.8 * RIGHT)
        fourth_line.to_edge(LEFT)
        fourth_line.next_to(second_line, 0.5 * DOWN)
        self.wait()
        self.play(GrowArrow(second_arrow))
        self.play(
            Transform(third_title, fourth_title),
            FadeOut(second_title),
            FadeInFrom(second_line, DOWN),
            FadeIn(third_line),
            FadeIn(fourth_line),
            FadeIn(fith_step_number),
        )

        sixth_step = VGroup(
            TextMobject("0"),
            TextMobject("1"),
            TextMobject("2"),
            TextMobject("3"),
            TextMobject("4"),
            TextMobject("5"),
            TextMobject("0"),
            TextMobject("6"),
            TextMobject("7"),
        ).arrange(DOWN, aligned_edge=LEFT)
        self.wait(2)
        self.play(
            FadeOut(fourth_line),
            FadeOut(third_line),
            FadeOut(second_line),
            FadeOut(fith_step_text),
            FadeOut(second_arrow),
            Transform(fith_step_number, sixth_step),
        )
        self.wait(2)
        seventh_step = VGroup(
            TextMobject("0"),
            TextMobject("1"),
            TextMobject("2"),
            TextMobject("3"),
            TextMobject("4"),
            TextMobject("5"),
            TextMobject("0"),
            TextMobject("6"),
            TextMobject("7"),
        ).arrange(2 * DOWN, aligned_edge=LEFT)

        seventh_step.next_to(sixth_step, 5 * LEFT)
        self.play(
            FadeOut(fith_step_number),
            FadeOut(third_title),
            FadeOut(fourth_title),
            Transform(sixth_step, seventh_step),
        )
        third_arrow = Arrow(RIGHT, 2 * RIGHT)
        third_arrow.next_to(seventh_step, RIGHT)
        embedding = torch.nn.Embedding(8, 4)
        # single data
        input_data = torch.LongTensor([[0, 1, 2, 3, 4, 5, 0, 6, 7]])
        # get the first image batch
        emdedding_matrix = (
            embedding(input_data).detach().numpy()[0].round(decimals=2).astype("str")
        )
        matrix_first = Matrix(emdedding_matrix)
        matrix_first.next_to(third_arrow, RIGHT)
        fifth_title = TextMobject("Each integer becomes the index of a matrix*")
        second_line = TextMobject("*Again, notice that two words")
        third_line = TextMobject("the", color=RED)
        fourth_line = TextMobject(" were mapped to the same vector")
        fifth_title.scale(0.5)
        second_line.scale(0.5)
        third_line.scale(0.5)
        fourth_line.scale(0.5)
        fifth_title.to_corner(UP + LEFT)
        second_line.to_edge(LEFT)
        second_line.next_to(fifth_title, DOWN)
        third_line.next_to(second_line, 0.8 * RIGHT)
        fourth_line.to_edge(LEFT)
        fourth_line.next_to(second_line, 0.5 * DOWN)
        self.play(GrowArrow(third_arrow))
        self.play(
            FadeIn(matrix_first),
            FadeIn(fifth_title),
            FadeInFrom(second_line, DOWN),
            FadeIn(third_line),
            FadeIn(fourth_line),
        )
        matrix_second = Matrix(emdedding_matrix)
        self.wait(5)
        self.play(
            FadeOut(sixth_step),
            FadeOut(third_arrow),
            FadeOut(seventh_step),
            FadeOut(fourth_line),
            FadeOut(third_line),
            FadeOut(second_line),
            FadeOut(fifth_title),
        )
        last_data = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        final_matrix = Matrix(
            embedding(last_data).detach().numpy()[0].round(decimals=2).astype("str")
        )
        final_matrix.scale(0.9)
        index_matrix = VGroup(
            TextMobject("0"),
            TextMobject("1"),
            TextMobject("2"),
            TextMobject("3"),
            TextMobject("4"),
            TextMobject("5"),
            TextMobject("6"),
            TextMobject("7"),
        ).arrange(1.5 * DOWN, aligned_edge=LEFT)
        index_matrix.next_to(final_matrix, LEFT)
        self.play(FadeOutAndShift(matrix_first, RIGHT), FadeInFrom(final_matrix, LEFT))
        self.wait()

        text_data = VGroup(
            TextMobject("the"),
            TextMobject("quick"),
            TextMobject("brown"),
            TextMobject("fox"),
            TextMobject("jumps"),
            TextMobject("over"),
            TextMobject("lazy"),
            TextMobject("dog"),
        ).arrange(1.5 * DOWN, aligned_edge=LEFT)
        text_data.next_to(index_matrix, LEFT)
        final_title = TextMobject("In the end it is just a lookup table")
        final_title.to_edge(UP)
        final_title.scale(0.5)
        self.play(FadeIn(index_matrix), FadeIn(text_data), Write(final_title))
        self.wait(2)
        braces_one = Brace(final_matrix, RIGHT)
        braces_one_text = braces_one.get_text("Vocabulary")
        self.play(FadeInFrom(braces_one, RIGHT), FadeInFrom(braces_one_text, RIGHT))
        self.wait(1)
        braces_two = Brace(final_matrix, DOWN)
        braces_two_text = braces_two.get_text(
            "A hyperparameter. In this case, I have chosen 4"
        )
        braces_two_text.scale(0.5)
        self.play(FadeInFrom(braces_two, RIGHT), FadeInFrom(braces_two_text, RIGHT))
        self.wait(4)


class TransformerEncoderExample(Scene):
    def construct(self):
        initial_text = TextMobject("The quick brown fox jumps over the lazy dog!")
        first_step = TextMobject("Define some initial text")
        first_step.scale(0.5)
        first_step.to_corner(UP + LEFT)
        self.play(FadeIn(first_step), Write(initial_text))
        self.wait(2)
        second_step = VGroup(
            TextMobject("Apply some tokenizer"),
            TextMobject("Here, I have decided to use WordPiece", color=BLUE),
        ).arrange(DOWN, aligned_edge=DOWN)
        second_step.scale(0.5)
        second_step.to_corner(UP + LEFT)
        second_text = TextMobject(
            "[",
            "[CLS]",
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
            "!",
            "[SEP]",
            "]",
        )
        second_text.arrange(RIGHT, buff=0.3)
        self.play(
            ReplacementTransform(first_step, second_step),
            ReplacementTransform(initial_text, second_text),
        )
        self.wait(2)
        obs_text = TextMobject("Notice that CLS and SEP are special tokens")
        obs_text.scale(0.5)
        obs_text.next_to(second_step, DOWN)
        obs_text_two = VGroup(
            TextMobject("CLS denotes the beginning of a sentence"),
            TextMobject("SEP inicates the end of it"),
        ).arrange(DOWN, aligned_edge=DOWN)
        obs_text_two.scale(0.5)
        obs_text_two.next_to(obs_text, DOWN)
        self.play(
            ApplyMethod(second_text[1].set_color, RED),
            ApplyMethod(second_text[-2].set_color, RED),
            FadeIn(obs_text),
            Write(obs_text_two),
        )
        self.wait(2.5)
        third_text = TextMobject(
            "[",
            "101",
            "1996",
            "4248",
            "2829",
            "4419",
            "14523",
            "2058",
            "1996",
            "13971",
            "3899",
            "999",
            "102",
            "]",
        )
        third_text.arrange(RIGHT, buff=0.3)
        third_text.scale(0.7)
        third_step = VGroup(
            TextMobject("Map each word to an  Integer that will serve as an index"),
            TextMobject("of a lookup table with the size of vocabulary (Embedding)"),
            TextMobject("In the case of BERT, the vocab is around 30000", color=BLUE),
        ).arrange(DOWN, aligned_edge=DOWN)
        third_step.scale(0.5)
        third_step.to_corner(UP + LEFT)
        self.play(
            FadeOut(obs_text_two),
            FadeOut(obs_text),
            ReplacementTransform(second_text, third_text),
            ReplacementTransform(second_step, third_step),
        )
        self.wait(4)
        braces_one = Brace(third_text, DOWN)
        braces_one_text = braces_one.get_text("Sentence A")
        fourth_step = VGroup(
            TextMobject(
                "Given that this was one from two sampled sentences, mark it as belonging"
            ),
            TextMobject(
                "to either sentence A (first part) of a sentence or sentence B (second part)"
            ),
            TextMobject(
                "In the case of BERT base, the sum of tokens in sentence A $+$ the number of tokens in sentence B does not exceed $512$",
                color=RED,
            ),
        ).arrange(DOWN, aligned_edge=LEFT)
        fourth_step.scale(0.7)
        fourth_step.to_corner(UP + LEFT)
        self.play(
            ReplacementTransform(third_step, fourth_step),
            FadeInFrom(braces_one, DOWN),
            FadeInFrom(braces_one_text, DOWN),
        )
        self.wait(3)
        text = third_text.copy()
        text.scale(0.5)
        text.to_edge(DOWN)
        self.play(
            ReplacementTransform(third_text, text),
            FadeOut(braces_one),
            FadeOut(braces_one_text),
        )
        first_arrow = Arrow(UP, 2 * UP)
        first_arrow.next_to(text, UP)
        self.play(GrowArrow(first_arrow))
        input_embedding = TextMobject("Input Embedding")
        input_embedding.bg = SurroundingRectangle(
            input_embedding, fill_color=RED, color=WHITE, fill_opacity=0.5
        )
        input_embedding_group = VGroup(input_embedding, input_embedding.bg)
        input_embedding.scale(0.5)
        input_embedding.bg.scale(0.55)
        input_embedding_group.next_to(first_arrow, UP)
        fifth_step = VGroup(
            TextMobject("Map the indices to the embedding vectors. Here,"),
            TextMobject("two embedding vectors are beging summed", color=BLUE),
            TextMobject(
                "One embedding is obtained from vocab lookup,",
                color=RED,
            ),
            TextMobject("which has shape $|V| \\times d_{model}$"),
            TextMobject("In the case of BERT base, $|V|=30000$ and $d_{model}=768$"),
        ).arrange(DOWN, aligned_edge=LEFT)
        fifth_step.to_corner(UP + LEFT)
        fifth_step.scale(0.7)
        self.play(
            FadeIn(input_embedding_group), ReplacementTransform(fourth_step, fifth_step)
        )
        self.wait(4)
        fifth_step_b = VGroup(
            TextMobject(
                "The other is the embedding of either A sentences or B sentences, which are",
                color=RED,
            ),
            TextMobject(
                "learned thorugh NSP (explained in the next section).",
                color=RED,
            ),
            TextMobject("Here, $E_{\\text{\{A or B}\}} = 1 \\times d_{model}$")
        ).arrange(DOWN, aligned_edge=LEFT)
        fifth_step_b.scale(0.6)
        self.play(FadeIn(fifth_step_b))
        self.wait(7)
        step_six = VGroup(
            TextMobject("Sum the embedding obtained from Positional"),
            TexMobject(
                "\\text{Encoder, where }E_{\\text{PE}} = |\\text{Sentence}| \\times d_{model}"
            ),
            TextMobject("where $\\text{Sentence}$ is the number of tokens in"),
            TextMobject("both sentences A and B)"),
        ).arrange(DOWN, aligned_edge=LEFT)
        step_six.to_corner(UP + LEFT)
        step_six.scale(0.6)
        self.play(FadeOut(fifth_step_b))
        input_embedding_up_center = input_embedding.bg.get_corner(UP)
        second_arrow = Arrow(
            input_embedding_up_center + [0, -0.25, 0],
            input_embedding_up_center + [0, 0.5, 0],
        )
        self.play(GrowArrow(second_arrow))
        position_encoding = TexMobject("\oplus")
        position_encoding.next_to(second_arrow, 0.2 * UP)
        position_encoding.scale(0.7)
        position_encoding_text = VGroup(
            TextMobject("Positional"), TextMobject("Encoding")
        ).arrange(DOWN, aligned_edge=LEFT)
        position_encoding_text.scale(0.5)
        position_encoding_text.next_to(position_encoding, LEFT)
        self.play(
            FadeIn(position_encoding),
            FadeIn(position_encoding_text),
            ReplacementTransform(fifth_step, step_six),
        )
        self.wait(5)
        rectangle = Rectangle(height=4, width=4)
        rectangle.next_to(position_encoding, 1.2 * UP)
        step_seven = VGroup(
            TextMobject("Input encoder layer"),
        ).arrange(DOWN, aligned_edge=LEFT + DOWN)
        step_seven.to_corner(UP + LEFT)
        step_seven.scale(0.5)
        self.play(FadeIn(rectangle), ReplacementTransform(step_six, step_seven))
        bottom_corner_left = rectangle.get_corner(LEFT + DOWN)
        bottom_center = np.array([0, bottom_corner_left[1], 0])
        pe_center = position_encoding.get_center()
        third_arrow = Arrow(pe_center, (bottom_center + [0, 1, 0]))
        curved_arrow = CurvedArrow(
            third_arrow.get_center(), (bottom_center + [0.5, 0.8, 0]), angle=TAU / 4
        )
        curved_arrow_two = CurvedArrow(
            third_arrow.get_center(), (bottom_center + [-0.5, 0.8, 0]), angle=-TAU / 4
        )

        multi_head = VGroup(
            TextMobject("Multi-Head"), TextMobject("Attention")
        ).arrange(DOWN, aligned_edge=LEFT)
        multi_head.bg = SurroundingRectangle(
            multi_head, fill_color=ORANGE, color=WHITE, fill_opacity=0.5
        )
        multi_head_group = VGroup(multi_head, multi_head.bg)
        multi_head.scale(0.5)
        multi_head.bg.scale(0.55)
        multi_head_group.next_to(third_arrow, UP - [0, 0.4, 0])
        step_seven_b = VGroup(
            TextMobject("The first sublayer is a "),
            TextMobject("Multi-head attention Mechanism"),
        ).arrange(DOWN, aligned_edge=LEFT)
        step_seven_b.next_to(step_seven, DOWN)
        step_seven_b.scale(0.5)
        self.play(
            FadeIn(third_arrow),
            FadeIn(curved_arrow),
            FadeIn(curved_arrow_two),
            FadeIn(multi_head_group),
            FadeIn(step_seven_b),
        )
        self.wait(3)
        add_norm = TextMobject("Add \& Norm")
        add_norm.bg = SurroundingRectangle(
            add_norm, fill_color=YELLOW, color=WHITE, fill_opacity=0.5
        )
        add_norm_group = VGroup(add_norm, add_norm.bg)
        add_norm.scale(0.5)
        add_norm.bg.scale(0.55)
        add_norm_group.next_to(multi_head_group, UP - [0, 0.4, 0])
        bottom_add_norm_left = add_norm.bg.get_corner(LEFT + DOWN)
        bottom_add_norm_center = add_norm.bg.get_center()
        bottom_add_norm_center_down = np.array([0, bottom_add_norm_left[1], 0])
        bottom_add_norm_center_left = np.array(
            [bottom_add_norm_left[0], bottom_add_norm_center[1], 0]
        )
        bottom_multi_left = multi_head.bg.get_corner(LEFT + UP)
        bottom_multi_head_center = np.array([0, bottom_multi_left[1], 0])
        line = Line(bottom_multi_head_center, bottom_add_norm_center_down)
        curved_arrow_three = CurvedArrow(
            third_arrow.get_center() + [0, -0.2, 0],
            (bottom_add_norm_center_left),
            angle=-TAU / 2.75,
        )
        step_seven_c = TextMobject("Apply residual connection $+$ Layer Normalization")
        step_seven_c.scale(0.4)
        step_seven_c.to_corner(UP + LEFT)
        step_seven_c.next_to(step_seven, DOWN)
        self.play(
            FadeIn(line),
            FadeIn(add_norm_group),
            FadeIn(curved_arrow_three),
            ReplacementTransform(step_seven_b, step_seven_c),
        )
        self.wait(2)
        feed_forward = VGroup(TextMobject("Feed"), TextMobject("Forward")).arrange(
            DOWN, aligned_edge=DOWN
        )
        feed_forward.bg = SurroundingRectangle(
            feed_forward, fill_color=BLUE, color=WHITE, fill_opacity=0.5
        )
        feed_forward_group = VGroup(feed_forward, feed_forward.bg)
        feed_forward.scale(0.5)
        feed_forward.bg.scale(0.55)
        upper_add_norm_center = add_norm.bg.get_corner(UP)
        fourth_arrow = Arrow(upper_add_norm_center, upper_add_norm_center + [0, 0.4, 0])
        feed_forward_group.next_to(fourth_arrow, UP + [0, -0.4, 0])
        second_add_norm = add_norm_group.copy()
        second_add_norm.next_to(feed_forward_group, UP - [0, 0.4, 0])
        step_seven_d = VGroup(
            TextMobject("The second sublayer is a "),
            TextMobject("Fully connected Feed Forward NN"),
        ).arrange(DOWN, aligned_edge=LEFT)
        step_seven_d.scale(0.5)
        step_seven_d.to_corner(UP + LEFT)
        step_seven_d.next_to(step_seven, DOWN)
        self.play(GrowArrow(fourth_arrow))
        self.play(
            FadeIn(feed_forward_group), ReplacementTransform(step_seven_c, step_seven_d)
        )
        self.wait(3)
        second_bottom_add_norm_left = second_add_norm.get_corner(LEFT + DOWN)
        second_bottom_add_norm_center = second_add_norm.get_center()
        second_bottom_add_norm_center_down = np.array(
            [0, second_bottom_add_norm_left[1], 0]
        )
        second_bottom_add_norm_center_left = np.array(
            [second_bottom_add_norm_left[0], second_bottom_add_norm_center[1], 0]
        )
        bottom_fnn_left = feed_forward.bg.get_corner(LEFT + UP)
        bottom_fnn_head_center = np.array([0, bottom_fnn_left[1], 0])
        second_line = Line(bottom_fnn_head_center, second_bottom_add_norm_center_down)
        curved_arrow_four = CurvedArrow(
            fourth_arrow.get_center(),
            (second_bottom_add_norm_center_left),
            angle=-TAU / 4,
        )
        step_seven_e = TextMobject("Apply residual connection $+$ Layer Normalization")
        step_seven_e.scale(0.4)
        step_seven_e.to_corner(UP + LEFT)
        step_seven_e.next_to(step_seven, DOWN)
        self.play(GrowArrow(curved_arrow_four))
        self.play(
            FadeIn(second_add_norm),
            FadeIn(second_line),
            ReplacementTransform(step_seven_d, step_seven_e),
        )
        self.wait(2)
        second_upper_add_norm_center = second_add_norm.get_corner(UP)
        final_arrow = Arrow(
            second_upper_add_norm_center + [0, -0.25, 0],
            second_upper_add_norm_center + [0, 1, 0],
        )
        self.play(GrowArrow(final_arrow), FadeOut(step_seven_e))
        self.wait(2)
        encoder_layer = VGroup(TextMobject("Encoder Layer")).arrange(
            DOWN, aligned_edge=LEFT
        )
        encoder_layer.bg = SurroundingRectangle(
            encoder_layer, fill_color=YELLOW, color=WHITE, fill_opacity=0.5
        )
        encoder_layer_group = VGroup(encoder_layer, encoder_layer.bg)
        encoder_layer.scale(0.5)
        encoder_layer.bg.scale(0.55)
        third_arrow_b = Arrow(pe_center, (bottom_center + [0, 0.4, 0]))
        encoder_layer_group.next_to(third_arrow_b, UP - [0, 0.4, 0])
        self.play(
            FadeOut(line),
            FadeOut(second_line),
            FadeOut(curved_arrow),
            FadeOut(curved_arrow_two),
            FadeOut(curved_arrow_three),
            FadeOut(curved_arrow_four),
            FadeOut(feed_forward_group),
            FadeOut(multi_head_group),
            FadeOut(add_norm_group),
            FadeOut(second_add_norm),
            FadeOut(final_arrow),
            FadeOut(fourth_arrow),
            ReplacementTransform(rectangle, encoder_layer_group),
            ReplacementTransform(third_arrow, third_arrow_b),
        )
        self.wait(3)
        step_eight = VGroup(
            TextMobject("Stack $N$ encoders"),
        ).arrange(DOWN, aligned_edge=LEFT + DOWN)
        step_eight.to_corner(UP + LEFT)
        step_eight.scale(0.5)
        encoder_layer_center = encoder_layer.get_corner(UP)
        arrow = Arrow(encoder_layer_center, encoder_layer_center + [0, 0.3, 0])
        self.play(ReplacementTransform(step_seven, step_eight), GrowArrow(arrow))
        second_encoder_layer = encoder_layer_group.copy()
        second_encoder_layer.next_to(arrow, UP + [0, -0.4, 0])
        second_encoder_layer_center = second_encoder_layer.get_corner(UP)
        second_arrow = Arrow(
            second_encoder_layer_center, second_encoder_layer_center + [0, 0.3, 0]
        )
        self.play(FadeIn(second_encoder_layer), FadeIn(second_arrow))
        third_encoder_layer = second_encoder_layer.copy()
        third_encoder_layer.next_to(second_arrow, UP + [0, -0.4, 0])
        third_encoder_layer_center = third_encoder_layer.get_corner(UP)
        third_arrow = Arrow(
            third_encoder_layer_center, third_encoder_layer_center + [0, 0.3, 0]
        )
        self.play(FadeIn(third_encoder_layer), FadeIn(third_arrow))
        vertical_dot = TextMobject("$\\vdots$")
        step_eight_b = TextMobject("In the case of BERT base, $N=12$")
        step_eight_b.scale(0.5)
        step_eight_b.to_corner(UP + LEFT)
        step_eight_b.next_to(step_eight, DOWN)
        vertical_dot.next_to(third_encoder_layer, UP)
        self.play(Write(vertical_dot), FadeIn(step_eight_b))
        self.wait(2)
        fourth_encoder_layer = third_encoder_layer.copy()
        fourth_encoder_layer.next_to(vertical_dot, UP + [0, -0.4, 0])
        fourth_encoder_layer_center = fourth_encoder_layer.get_corner(UP)
        fourth_arrow = Arrow(
            fourth_encoder_layer_center, fourth_encoder_layer_center + [0, 0.3, 0]
        )
        self.play(FadeIn(fourth_encoder_layer), FadeIn(fourth_arrow))
        final_text = TextMobject(
            "[",
            "$E_{\\text{[CLS]}}$",
            "$E_{\\text{the}}$",
            "$E_{\\text{quick}}$",
            "$E_{\\text{brown}}$",
            "$E_{\\text{fox}}$",
            "$E_{\\text{jumps}}$",
            "$E_{\\text{over}}$",
            "$E_{\\text{the}}$",
            "$E_{\\text{lazy}}$",
            "$E_{\\text{dog}}$",
            "$E_{\\text{!}}$",
            "$E_{\\text{[SEP]}}$",
            "]",
        )
        final_text.arrange(RIGHT, buff=0.3)
        final_text.scale(0.5)
        final_text.next_to(fourth_arrow, UP)
        step_nine = TextMobject("Each word has a Embedding vector of size $d_{model}$")
        step_nine.scale(0.5)
        step_nine.to_corner(UP+LEFT)
        self.play(
            FadeIn(final_text),
            FadeOut(step_eight_b),
            ReplacementTransform(step_eight, step_nine),
        )
        self.wait(4)
