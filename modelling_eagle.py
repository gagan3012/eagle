from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    PreTrainedModel, Swinv2Model, AutoModel, GPTNeoForCausalLM

from .configuration_eagle import EagleConfig

from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput



class EagleModelForCausalLMModelOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    connector_outputs: Optional[Tuple[torch.FloatTensor]] = None
    lm_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "connector_outputs", "lm_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class EaglePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = EagleConfig
    base_model_prefix = "Eagle"
    supports_gradient_checkpointing = True


class EagleVisionModel(EaglePreTrainedModel):
    main_input_name= "pixel_values"

    config_class = EagleConfig

    def __init__(self, config: EagleConfig):
        super().__init__(config)

        self.vision_encoder = Swinv2Model.from_pretrained(config.vision_encoder_model_name_or_path)
        self.vision_config = self.vision_encoder.config
        self.post_layer_norm = nn.LayerNorm(self.vision_config.hidden_size, 
                                            eps=self.vision_config.layer_norm_eps)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, Tuple[torch.Tensor, ...]]:
        
        encoder_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layer_norm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
class EagleConnectorModel(EaglePreTrainedModel):

    def __init__(self, config: EagleConfig):
        super().__init__(config)

        self.connector_config = AutoConfig.from_pretrained(config.connector_model_name_or_path)
        
        self.connector_config.encoder_width = config.vision_hidden_size
        self.connector_config.is_decoder = True
        self.connector_config.add_cross_attention = True
        self.connector_config.cross_attention_freq = config.cross_attention_freq
        self.connector_config.query_length = config.query_length

        self.layernorm = nn.LayerNorm(self.connector_config.hidden_size, eps=self.connector_config.layer_norm_eps)
        self.dropout = nn.Dropout(self.connector_config.hidden_dropout_prob)

        self.connector_model = AutoModel.from_pretrained(config.connector_model_name_or_path, config=self.connector_config)

        query_tokens = nn.Parameter(torch.zeros(1, config.query_length, config.vision_hidden_size))

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_tokens,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_tokens.shape[1] if query_tokens is not None else 0

        embedding_output = self.layernorm(query_tokens)
        embedding_output = self.dropout(embedding_output)

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.connector_config.num_hidden_layers)

        print(attention_mask.shape)

        encoder_outputs = self.connector_model.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    

class EagleModel(EaglePreTrainedModel):
    config_class = EagleConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EagleConfig):
        super().__init__(config)

        self.vision_model = EagleVisionModel(config)

        self.connector_model = EagleConnectorModel(config)

        self.connector_config =  AutoConfig.from_pretrained(config.connector_model_name_or_path)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, self.connector_config.hidden_size))

        self.language_model = GPTNeoForCausalLM.from_pretrained(config.language_model_name_or_path)

        self.connector_to_lm = nn.Linear(self.connector_config.hidden_size,
                                         self.language_model.config.hidden_size, bias=False)
        
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vision_sequence_output = vision_outputs[0]

        image_attention_mask = torch.ones(vision_sequence_output.size()[:-1], dtype=torch.long, 
                                          device=vision_sequence_output.device)
        
        query_tokens = self.query_tokens.expand(vision_sequence_output.shape[0], -1, -1)
              
        connector_outputs = self.connector_model(
            query_tokens=query_tokens,
            attention_mask=image_attention_mask,
            encoder_hidden_states=vision_sequence_output,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        connector_sequence_output = connector_outputs[0]

        lm_sequence_output = self.connector_to_lm(connector_sequence_output)

        lm_attention_mask = torch.ones(lm_sequence_output.size()[:-1], dtype=torch.long,
                                        device=lm_sequence_output.device)
        
        input_embeds = self.language_model.get_input_embeddings(input_ids=input_ids)

        input_embeds = torch.cat([input_embeds, lm_sequence_output], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_embeds, dtype=torch.long, 
                                        device=input_embeds.device)
        device = lm_attention_mask.device

        attention_mask = torch.cat([attention_mask, lm_attention_mask], dim=1)

        lm_outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = lm_outputs.logits if return_dict else lm_outputs[1]

        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, vision_outputs,
                      connector_outputs, lm_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return EagleModelForCausalLMModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            connector_outputs=connector_outputs,
            lm_outputs=lm_outputs,
        )


class EagleModelForCausalLM(EaglePreTrainedModel):
    config_class = EagleConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EagleConfig):
        super().__init__(config)

        self.vision_model = EagleVisionModel(config)

        self.connector_model = EagleConnectorModel(config)

        self.connector_config = AutoConfig.from_pretrained(
            config.connector_model_name_or_path)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, self.connector_config.hidden_size))

        self.language_model = GPTNeoForCausalLM.from_pretrained(
            config.language_model_name_or_path)

        self.connector_to_lm = nn.Linear(self.connector_config.hidden_size,
                                         self.language_model.config.hidden_size, bias=False)
        
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vision_sequence_output = vision_outputs[0]

        image_attention_mask = torch.ones(vision_sequence_output.size()[:-1], dtype=torch.long, 
                                          device=vision_sequence_output.device)

        query_tokens = self.query_tokens.expand(vision_sequence_output.shape[0], -1, -1)

        connector_outputs = self.connector_model(
            query_tokens=query_tokens,
            attention_mask=image_attention_mask,
            encoder_hidden_states=vision_sequence_output,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        connector_sequence_output = connector_outputs[0]

        lm_sequence_input = self.connector_to_lm(connector_sequence_output)

        lm_attention_mask = torch.ones(lm_sequence_input.size()[:-1], dtype=torch.long,
                                        device=lm_sequence_input.device)
        
        input_embeds = self.language_model.get_input_embeddings(input_ids=input_ids)

        input_embeds = torch.cat([input_embeds, lm_sequence_input], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_embeds, dtype=torch.long, 
                                        device=input_embeds.device)
        device = lm_attention_mask.device

        attention_mask = torch.cat([attention_mask, lm_attention_mask], dim=1, device=device)

        lm_outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = lm_outputs.logits if return_dict else lm_outputs[1]

        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, vision_outputs,
                      connector_outputs, lm_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return EagleModelForCausalLMModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            connector_outputs=connector_outputs,
            lm_outputs=lm_outputs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    )-> torch.LongTensor:
        
        batch_size = pixel_values.shape[0]

        image_embeds = self.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long,
                                            device=image_embeds.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        connector_outputs = self.connector_model(
            query_tokens=query_tokens,
            attention_mask=image_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )

        connector_sequence_output = connector_outputs.last_hidden_state

        lm_sequence_input = self.connector_to_lm(connector_sequence_output)

        lm_attention_mask = torch.ones(lm_sequence_input.size()[:-1], dtype=torch.long,
                                        device=lm_sequence_input.device)
        
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.language_model.config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([lm_attention_mask, attention_mask.to(lm_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([lm_sequence_input, inputs_embeds.to(lm_sequence_input.device)], dim=1)

        output_sequences = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return output_sequences
