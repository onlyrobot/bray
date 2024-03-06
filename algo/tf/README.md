# encoder
负责处理每一块结构化输入
假设一个moba项目, 它神经网络的输入由3个模块组成:
* 英雄信息向量  common_vector shape:(16,)
* 小兵单位向量  solider_vec  shape:(20, 12)  假设小兵数量的最大上限为20, 每个小兵的特征长度为12
* 英雄局部视野  local map     shape:(32, 32, 17)
针对以上3个模块的输入，需要使用3个encoder
```python
common_vec_encoder = CommonEncoder(
    input_name="common_vector",
    output_name="common_vector",
    hidden_layer_sizes=[128]
)

solider_vec_encoder = EntityEncoder(
    input_name="solider_vec",
    output_name="solider_vec",
    hidden_layer_sizes=[128]
)

local_map_encoder = SpatialEncoder(
    input_name="local_map",
    output_name="local_map",
    projected_channel_num=64,
    output_size=512,
    down_samples=[
        (64, 4, 2, 'same', True),
        (128, 4, 2, 'same', False),
    ],
)
```
# aggregator
负责将所有encoder的输出concat后统一处理, 默认使用DenseAggragator

# decoder
负责处理每一个独立的动作头
假设一个moba项目, 它的动作空间组成如下
* 主动作  
&nbsp;&nbsp;main_decision  
&nbsp;&nbsp;类型:离散动作  
&nbsp;&nbsp;维度:4  
&nbsp;&nbsp;备注:停止, 移动, 普攻, 释放技能, 技能目标  
* 移动  
&nbsp;&nbsp;move_direction  
&nbsp;&nbsp;类型:离散动作  
&nbsp;&nbsp;维度:8  
* 普攻  
&nbsp;&nbsp;attack_target  
&nbsp;&nbsp;类型:目标选择  
* 释放技能  
&nbsp;&nbsp;select_skill  
&nbsp;&nbsp;类型:目标选择  
* 技能目标  
&nbsp;&nbsp;skill_target  
&nbsp;&nbsp;类型:目标选择  

针对以上动作空间设计, 需要使用5个decoder
注: 每个decoder均需要搭配一个action head对象

```python
main_decision_cmd_decoder = CategoricalDecoder(
    n=4,
    hidden_layer_sizes=[],
    temperature=decoder_temperature,
    )
main_decision_action_head = ActionHead(
    "main_decision",
    mask_feature_name="main_decision_action_mask",
    source_feature_name=None,
    auto_regressive=False,
    dependency="origin"
)

move_cmd_decoder = CategoricalDecoder(
    n=8,
    hidden_layer_sizes=[],
    temperature=decoder_temperature,
    )
move_action_head = ActionHead(
    "move_direction",
    mask_feature_name="move_direction_action_mask",
    source_feature_name=None,
    auto_regressive=False,
    dependency="origin"
)

attack_target_cmd_decoder = SingleSelectiveDecoder(
    temperature=decoder_temperature,
    )
attack_target_action_head = ActionHead(
    "attack_target",
    mask_feature_name="attack_target_action_mask",
    source_feature_name='attackable_unit_embeddings',
    auto_regressive=False,
    dependency="origin"
)

select_skill_cmd_decoder = SingleSelectiveDecoder(
    temperature=decoder_temperature,
    auto_regressive_method='plus'
    )
select_skill_action_head = ActionHead(
    "select_skill",
    mask_feature_name="select_skill_action_mask",
    source_feature_name='skill_embeddings',
    auto_regressive=True,
    dependency="origin"
)

skill_target_cmd_decoder = SingleSelectiveDecoder(
    temperature=decoder_temperature,
    )
skill_target_action_head = ActionHead(
    "skill_target",
    mask_feature_name="skill_target_action_mask",
    source_feature_name='skillable_unit_embeddings',
    auto_regressive=False,
)
```
