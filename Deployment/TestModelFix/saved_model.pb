ÉŁ/
°
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ľ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint˙˙˙˙˙˙˙˙˙
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38ÂŃ-

 Adam/lstm_16/lstm_cell_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_16/lstm_cell_16/bias/v

4Adam/lstm_16/lstm_cell_16/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_16/lstm_cell_16/bias/v*
_output_shapes	
:*
dtype0
ľ
,Adam/lstm_16/lstm_cell_16/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v
Ž
@Adam/lstm_16/lstm_cell_16/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v*
_output_shapes
:	 *
dtype0
Ą
"Adam/lstm_16/lstm_cell_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *3
shared_name$"Adam/lstm_16/lstm_cell_16/kernel/v

6Adam/lstm_16/lstm_cell_16/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_16/lstm_cell_16/kernel/v*
_output_shapes
:	 *
dtype0

Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:*
dtype0

Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_16/kernel/v

*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

: *
dtype0

Adam/embedding_16/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	' */
shared_name Adam/embedding_16/embeddings/v

2Adam/embedding_16/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_16/embeddings/v*
_output_shapes
:	' *
dtype0

 Adam/lstm_16/lstm_cell_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_16/lstm_cell_16/bias/m

4Adam/lstm_16/lstm_cell_16/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_16/lstm_cell_16/bias/m*
_output_shapes	
:*
dtype0
ľ
,Adam/lstm_16/lstm_cell_16/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *=
shared_name.,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m
Ž
@Adam/lstm_16/lstm_cell_16/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m*
_output_shapes
:	 *
dtype0
Ą
"Adam/lstm_16/lstm_cell_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *3
shared_name$"Adam/lstm_16/lstm_cell_16/kernel/m

6Adam/lstm_16/lstm_cell_16/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_16/lstm_cell_16/kernel/m*
_output_shapes
:	 *
dtype0

Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:*
dtype0

Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_16/kernel/m

*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

: *
dtype0

Adam/embedding_16/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	' */
shared_name Adam/embedding_16/embeddings/m

2Adam/embedding_16/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_16/embeddings/m*
_output_shapes
:	' *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

lstm_16/lstm_cell_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_16/lstm_cell_16/bias

-lstm_16/lstm_cell_16/bias/Read/ReadVariableOpReadVariableOplstm_16/lstm_cell_16/bias*
_output_shapes	
:*
dtype0
§
%lstm_16/lstm_cell_16/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *6
shared_name'%lstm_16/lstm_cell_16/recurrent_kernel
 
9lstm_16/lstm_cell_16/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_16/lstm_cell_16/recurrent_kernel*
_output_shapes
:	 *
dtype0

lstm_16/lstm_cell_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *,
shared_namelstm_16/lstm_cell_16/kernel

/lstm_16/lstm_cell_16/kernel/Read/ReadVariableOpReadVariableOplstm_16/lstm_cell_16/kernel*
_output_shapes
:	 *
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

: *
dtype0

embedding_16/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	' *(
shared_nameembedding_16/embeddings

+embedding_16/embeddings/Read/ReadVariableOpReadVariableOpembedding_16/embeddings*
_output_shapes
:	' *
dtype0

NoOpNoOp
đ9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ť9
valueĄ9B9 B9
Î
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
Ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
Á
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator
#cell
$
state_spec*
Ś
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
.
0
-1
.2
/3
+4
,5*
.
0
-1
.2
/3
+4
,5*
* 
°
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
5trace_0
6trace_1
7trace_2
8trace_3* 
6
9trace_0
:trace_1
;trace_2
<trace_3* 
* 
ź
=iter

>beta_1

?beta_2
	@decay
Alearning_ratem+m,m-m.m/mv+v,v-v.v/v*

Bserving_default* 

0*

0*
* 

Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Htrace_0* 

Itrace_0* 
ke
VARIABLE_VALUEembedding_16/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Otrace_0
Ptrace_1* 

Qtrace_0
Rtrace_1* 
* 

-0
.1
/2*

-0
.1
/2*
* 


Sstates
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
6
Ytrace_0
Ztrace_1
[trace_2
\trace_3* 
6
]trace_0
^trace_1
_trace_2
`trace_3* 
* 
ă
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator
h
state_size

-kernel
.recurrent_kernel
/bias*
* 

+0
,1*

+0
,1*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
_Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_16/lstm_cell_16/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_16/lstm_cell_16/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_16/lstm_cell_16/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

p0
q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

#0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1
/2*

-0
.1
/2*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
w	variables
x	keras_api
	ytotal
	zcount*
H
{	variables
|	keras_api
	}total
	~count

_fn_kwargs*
* 
* 
* 
* 
* 

y0
z1*

w	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

}0
~1*

{	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUEAdam/embedding_16/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_16/lstm_cell_16/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_16/lstm_cell_16/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_16/lstm_cell_16/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/embedding_16/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_16/lstm_cell_16/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_16/lstm_cell_16/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_16/lstm_cell_16/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

"serving_default_embedding_16_inputPlaceholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
á
StatefulPartitionedCallStatefulPartitionedCall"serving_default_embedding_16_inputembedding_16/embeddingslstm_16/lstm_cell_16/kernel%lstm_16/lstm_cell_16/recurrent_kernellstm_16/lstm_cell_16/biasdense_16/kerneldense_16/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_204208
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_16/embeddings/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp/lstm_16/lstm_cell_16/kernel/Read/ReadVariableOp9lstm_16/lstm_cell_16/recurrent_kernel/Read/ReadVariableOp-lstm_16/lstm_cell_16/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/embedding_16/embeddings/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp6Adam/lstm_16/lstm_cell_16/kernel/m/Read/ReadVariableOp@Adam/lstm_16/lstm_cell_16/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_16/lstm_cell_16/bias/m/Read/ReadVariableOp2Adam/embedding_16/embeddings/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp6Adam/lstm_16/lstm_cell_16/kernel/v/Read/ReadVariableOp@Adam/lstm_16/lstm_cell_16/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_16/lstm_cell_16/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_207063
˛
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_16/embeddingsdense_16/kerneldense_16/biaslstm_16/lstm_cell_16/kernel%lstm_16/lstm_cell_16/recurrent_kernellstm_16/lstm_cell_16/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/embedding_16/embeddings/mAdam/dense_16/kernel/mAdam/dense_16/bias/m"Adam/lstm_16/lstm_cell_16/kernel/m,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m Adam/lstm_16/lstm_cell_16/bias/mAdam/embedding_16/embeddings/vAdam/dense_16/kernel/vAdam/dense_16/bias/v"Adam/lstm_16/lstm_cell_16/kernel/v,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v Adam/lstm_16/lstm_cell_16/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_207154ŁÇ,
§
d
+__inference_dropout_16_layer_call_fn_205162

inputs
identity˘StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_204061|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ôK
Ą
'__forward_gpu_lstm_with_fallback_206936

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d494a307-dced-4269-b473-1a192b8f604d*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_206761_206937*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_205902

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_31626796-2eeb-455b-b989-31f5c67391d1*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ź
Ł
I__inference_sequential_16_layer_call_and_return_conditional_losses_203565

inputs&
embedding_16_203101:	' !
lstm_16_203540:	 !
lstm_16_203542:	 
lstm_16_203544:	!
dense_16_203559: 
dense_16_203561:
identity˘ dense_16/StatefulPartitionedCall˘$embedding_16/StatefulPartitionedCall˘lstm_16/StatefulPartitionedCallö
$embedding_16/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_16_203101*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_16_layer_call_and_return_conditional_losses_203100đ
dropout_16/PartitionedCallPartitionedCall-embedding_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_203109
lstm_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0lstm_16_203540lstm_16_203542lstm_16_203544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_203539
 dense_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0dense_16_203559dense_16_203561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_203558x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
NoOpNoOp!^dense_16/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
§
˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_202634

inputs/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:˛
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_202361i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
	
Á
while_cond_205721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_205721___redundant_placeholder04
0while_while_cond_205721___redundant_placeholder14
0while_while_cond_205721___redundant_placeholder24
0while_while_cond_205721___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
Á
while_cond_203672
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_203672___redundant_placeholder04
0while_while_cond_203672___redundant_placeholder14
0while_while_cond_203672___redundant_placeholder24
0while_while_cond_203672___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
 

ő
D__inference_dense_16_layer_call_and_return_conditional_losses_203558

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ž;
Ŕ
 __inference_standard_lstm_204855

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_204769*
condR
while_cond_204768*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_9daf30b6-f686-449c-9bba-2a356f8c406f*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
	
Á
while_cond_204768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_204768___redundant_placeholder04
0while_while_cond_204768___redundant_placeholder14
0while_while_cond_204768___redundant_placeholder24
0while_while_cond_204768___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
	
Á
while_cond_206579
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_206579___redundant_placeholder04
0while_while_cond_206579___redundant_placeholder14
0while_while_cond_206579___redundant_placeholder24
0while_while_cond_206579___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ů)
Ď
while_body_203180
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
ž;
Ŕ
 __inference_standard_lstm_206666

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_206580*
condR
while_cond_206579*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d494a307-dced-4269-b473-1a192b8f604d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_206332_206508
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_2da92c77-19db-4d69-ae78-abd81b86f8ac*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_206507*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ĺ	
Ś
H__inference_embedding_16_layer_call_and_return_conditional_losses_203100

inputs*
embedding_lookup_203094:	' 
identity˘embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ä
embedding_lookupResourceGatherembedding_lookup_203094Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/203094*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0Ť
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/203094*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Á
while_cond_201834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_201834___redundant_placeholder04
0while_while_cond_201834___redundant_placeholder14
0while_while_cond_201834___redundant_placeholder24
0while_while_cond_201834___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ç@
Ě
)__inference_gpu_lstm_with_fallback_205473

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_a972921d-1e9d-4839-9033-309e1f33584b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
§
˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_206510

inputs/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:˛
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_206237i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
§
˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_203074

inputs/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:˛
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_202801i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ů)
Ď
while_body_202275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
	
Á
while_cond_204318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_204318___redundant_placeholder04
0while_while_cond_204318___redundant_placeholder14
0while_while_cond_204318___redundant_placeholder24
0while_while_cond_204318___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

ˇ
(__inference_lstm_16_layer_call_fn_205190
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
identity˘StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_202634o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
inputs/0
	
Á
while_cond_202274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_202274___redundant_placeholder04
0while_while_cond_202274___redundant_placeholder14
0while_while_cond_202274___redundant_placeholder24
0while_while_cond_202274___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ž;
Ŕ
 __inference_standard_lstm_202801

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_202715*
condR
while_cond_202714*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_86580efb-f15a-4581-9517-8d7c58130198*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_202896_203072
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_86580efb-f15a-4581-9517-8d7c58130198*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_203071*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ôK
Ą
'__forward_gpu_lstm_with_fallback_205125

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_9daf30b6-f686-449c-9bba-2a356f8c406f*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_204950_205126*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_203360

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_f43cc504-968b-46c4-93a8-e6ba5dafc816*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
˛	

.__inference_sequential_16_layer_call_fn_204143
embedding_16_input
unknown:	' 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_204111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
,
_user_specified_nameembedding_16_input
§
˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_203539

inputs/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:˛
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_203266i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ů)
Ď
while_body_202715
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
§
˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_206939

inputs/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:˛
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_206666i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ç@
Ě
)__inference_gpu_lstm_with_fallback_206331

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_2da92c77-19db-4d69-ae78-abd81b86f8ac*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
 

ő
D__inference_dense_16_layer_call_and_return_conditional_losses_206959

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
	
Á
while_cond_203179
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_203179___redundant_placeholder04
0while_while_cond_203179___redundant_placeholder14
0while_while_cond_203179___redundant_placeholder24
0while_while_cond_203179___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ôK
Ą
'__forward_gpu_lstm_with_fallback_203536

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_f43cc504-968b-46c4-93a8-e6ba5dafc816*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_203361_203537*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_202455

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d8412ebb-b775-4f9b-a70b-ce15dad8ece0*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ž;
Ŕ
 __inference_standard_lstm_204405

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_204319*
condR
while_cond_204318*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_912539c9-9430-40df-bd9c-e69ba5214b52*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Â

)__inference_dense_16_layer_call_fn_206948

inputs
unknown: 
	unknown_0:
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_203558o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ôK
Ą
'__forward_gpu_lstm_with_fallback_203071

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_86580efb-f15a-4581-9517-8d7c58130198*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_202896_203072*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ţ.

I__inference_sequential_16_layer_call_and_return_conditional_losses_204685

inputs7
$embedding_16_embedding_lookup_204246:	' 7
$lstm_16_read_readvariableop_resource:	 9
&lstm_16_read_1_readvariableop_resource:	 5
&lstm_16_read_2_readvariableop_resource:	9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:
identity˘dense_16/BiasAdd/ReadVariableOp˘dense_16/MatMul/ReadVariableOp˘embedding_16/embedding_lookup˘lstm_16/Read/ReadVariableOp˘lstm_16/Read_1/ReadVariableOp˘lstm_16/Read_2/ReadVariableOpk
embedding_16/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ř
embedding_16/embedding_lookupResourceGather$embedding_16_embedding_lookup_204246embedding_16/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_16/embedding_lookup/204246*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0Ň
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_16/embedding_lookup/204246*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ¤
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
dropout_16/IdentityIdentity1embedding_16/embedding_lookup/Identity_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ Y
lstm_16/ShapeShapedropout_16/Identity:output:0*
T0*
_output_shapes
:e
lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ů
lstm_16/strided_sliceStridedSlicelstm_16/Shape:output:0$lstm_16/strided_slice/stack:output:0&lstm_16/strided_slice/stack_1:output:0&lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_16/zeros/packedPacklstm_16/strided_slice:output:0lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_16/zerosFilllstm_16/zeros/packed:output:0lstm_16/zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z
lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_16/zeros_1/packedPacklstm_16/strided_slice:output:0!lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_16/zeros_1Filllstm_16/zeros_1/packed:output:0lstm_16/zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
lstm_16/Read/ReadVariableOpReadVariableOp$lstm_16_read_readvariableop_resource*
_output_shapes
:	 *
dtype0k
lstm_16/IdentityIdentity#lstm_16/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
lstm_16/Read_1/ReadVariableOpReadVariableOp&lstm_16_read_1_readvariableop_resource*
_output_shapes
:	 *
dtype0o
lstm_16/Identity_1Identity%lstm_16/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
lstm_16/Read_2/ReadVariableOpReadVariableOp&lstm_16_read_2_readvariableop_resource*
_output_shapes	
:*
dtype0k
lstm_16/Identity_2Identity%lstm_16/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ř
lstm_16/PartitionedCallPartitionedCalldropout_16/Identity:output:0lstm_16/zeros:output:0lstm_16/zeros_1:output:0lstm_16/Identity:output:0lstm_16/Identity_1:output:0lstm_16/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_204405
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_16/MatMulMatMul lstm_16/PartitionedCall:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
dense_16/SoftmaxSoftmaxdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentitydense_16/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp^embedding_16/embedding_lookup^lstm_16/Read/ReadVariableOp^lstm_16/Read_1/ReadVariableOp^lstm_16/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2:
lstm_16/Read/ReadVariableOplstm_16/Read/ReadVariableOp2>
lstm_16/Read_1/ReadVariableOplstm_16/Read_1/ReadVariableOp2>
lstm_16/Read_2/ReadVariableOplstm_16/Read_2/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ôK
Ą
'__forward_gpu_lstm_with_fallback_204029

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_b63e776e-adf7-4312-a3b9-4155f169d032*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_203854_204030*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_204499

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_912539c9-9430-40df-bd9c-e69ba5214b52*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ż
ż
C__inference_lstm_16_layer_call_and_return_conditional_losses_205652
inputs_0/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:´
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_205379i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
inputs/0
ž;
Ŕ
 __inference_standard_lstm_203266

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_203180*
condR
while_cond_203179*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_f43cc504-968b-46c4-93a8-e6ba5dafc816*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ő
G
+__inference_dropout_16_layer_call_fn_205157

inputs
identityž
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_203109m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
č

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_205179

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @q
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ľ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ł
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ |
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ v
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ f
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_206761_206937
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_d494a307-dced-4269-b473-1a192b8f604d*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_206936*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
Á
while_cond_206150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_206150___redundant_placeholder04
0while_while_cond_206150___redundant_placeholder14
0while_while_cond_206150___redundant_placeholder24
0while_while_cond_206150___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ü
Ô
I__inference_sequential_16_layer_call_and_return_conditional_losses_204183
embedding_16_input&
embedding_16_204166:	' !
lstm_16_204170:	 !
lstm_16_204172:	 
lstm_16_204174:	!
dense_16_204177: 
dense_16_204179:
identity˘ dense_16/StatefulPartitionedCall˘"dropout_16/StatefulPartitionedCall˘$embedding_16/StatefulPartitionedCall˘lstm_16/StatefulPartitionedCall
$embedding_16/StatefulPartitionedCallStatefulPartitionedCallembedding_16_inputembedding_16_204166*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_16_layer_call_and_return_conditional_losses_203100
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall-embedding_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_204061Ł
lstm_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0lstm_16_204170lstm_16_204172lstm_16_204174*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_204032
 dense_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0dense_16_204177dense_16_204179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_203558x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙×
NoOpNoOp!^dense_16/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall:d `
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
,
_user_specified_nameembedding_16_input
Đ
Ż
I__inference_sequential_16_layer_call_and_return_conditional_losses_204163
embedding_16_input&
embedding_16_204146:	' !
lstm_16_204150:	 !
lstm_16_204152:	 
lstm_16_204154:	!
dense_16_204157: 
dense_16_204159:
identity˘ dense_16/StatefulPartitionedCall˘$embedding_16/StatefulPartitionedCall˘lstm_16/StatefulPartitionedCall
$embedding_16/StatefulPartitionedCallStatefulPartitionedCallembedding_16_inputembedding_16_204146*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_16_layer_call_and_return_conditional_losses_203100đ
dropout_16/PartitionedCallPartitionedCall-embedding_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_203109
lstm_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0lstm_16_204150lstm_16_204152lstm_16_204154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_203539
 dense_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0dense_16_204157dense_16_204159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_203558x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
NoOpNoOp!^dense_16/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall:d `
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
,
_user_specified_nameembedding_16_input
Ż
ż
C__inference_lstm_16_layer_call_and_return_conditional_losses_206081
inputs_0/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:´
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_205808i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
inputs/0
Ľ7

I__inference_sequential_16_layer_call_and_return_conditional_losses_205135

inputs7
$embedding_16_embedding_lookup_204689:	' 7
$lstm_16_read_readvariableop_resource:	 9
&lstm_16_read_1_readvariableop_resource:	 5
&lstm_16_read_2_readvariableop_resource:	9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:
identity˘dense_16/BiasAdd/ReadVariableOp˘dense_16/MatMul/ReadVariableOp˘embedding_16/embedding_lookup˘lstm_16/Read/ReadVariableOp˘lstm_16/Read_1/ReadVariableOp˘lstm_16/Read_2/ReadVariableOpk
embedding_16/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ř
embedding_16/embedding_lookupResourceGather$embedding_16_embedding_lookup_204689embedding_16/Cast:y:0*
Tindices0*7
_class-
+)loc:@embedding_16/embedding_lookup/204689*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0Ň
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_16/embedding_lookup/204689*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ¤
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @˛
dropout_16/dropout/MulMul1embedding_16/embedding_lookup/Identity_1:output:0!dropout_16/dropout/Const:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ y
dropout_16/dropout/ShapeShape1embedding_16/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:ť
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0*

seedf
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ô
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ Y
lstm_16/ShapeShapedropout_16/dropout/Mul_1:z:0*
T0*
_output_shapes
:e
lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ů
lstm_16/strided_sliceStridedSlicelstm_16/Shape:output:0$lstm_16/strided_slice/stack:output:0&lstm_16/strided_slice/stack_1:output:0&lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_16/zeros/packedPacklstm_16/strided_slice:output:0lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_16/zerosFilllstm_16/zeros/packed:output:0lstm_16/zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z
lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
lstm_16/zeros_1/packedPacklstm_16/strided_slice:output:0!lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_16/zeros_1Filllstm_16/zeros_1/packed:output:0lstm_16/zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
lstm_16/Read/ReadVariableOpReadVariableOp$lstm_16_read_readvariableop_resource*
_output_shapes
:	 *
dtype0k
lstm_16/IdentityIdentity#lstm_16/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
lstm_16/Read_1/ReadVariableOpReadVariableOp&lstm_16_read_1_readvariableop_resource*
_output_shapes
:	 *
dtype0o
lstm_16/Identity_1Identity%lstm_16/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
lstm_16/Read_2/ReadVariableOpReadVariableOp&lstm_16_read_2_readvariableop_resource*
_output_shapes	
:*
dtype0k
lstm_16/Identity_2Identity%lstm_16/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ř
lstm_16/PartitionedCallPartitionedCalldropout_16/dropout/Mul_1:z:0lstm_16/zeros:output:0lstm_16/zeros_1:output:0lstm_16/Identity:output:0lstm_16/Identity_1:output:0lstm_16/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_204855
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_16/MatMulMatMul lstm_16/PartitionedCall:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
dense_16/SoftmaxSoftmaxdense_16/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
IdentityIdentitydense_16/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp^embedding_16/embedding_lookup^lstm_16/Read/ReadVariableOp^lstm_16/Read_1/ReadVariableOp^lstm_16/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2:
lstm_16/Read/ReadVariableOplstm_16/Read/ReadVariableOp2>
lstm_16/Read_1/ReadVariableOplstm_16/Read_1/ReadVariableOp2>
lstm_16/Read_2/ReadVariableOplstm_16/Read_2/ReadVariableOp:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
Á
while_cond_205292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_205292___redundant_placeholder04
0while_while_cond_205292___redundant_placeholder14
0while_while_cond_205292___redundant_placeholder24
0while_while_cond_205292___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ů)
Ď
while_body_205293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_205474_205650
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_a972921d-1e9d-4839-9033-309e1f33584b*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_205649*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	

$__inference_signature_wrapper_204208
embedding_16_input
unknown:	' 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallembedding_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_202201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
,
_user_specified_nameembedding_16_input

ľ
(__inference_lstm_16_layer_call_fn_205212

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_203539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ů)
Ď
while_body_203673
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
Ů)
Ď
while_body_205722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
Ç@
Ě
)__inference_gpu_lstm_with_fallback_204949

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_9daf30b6-f686-449c-9bba-2a356f8c406f*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_204950_205126
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_9daf30b6-f686-449c-9bba-2a356f8c406f*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_205125*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ôK
Ą
'__forward_gpu_lstm_with_fallback_205649

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_a972921d-1e9d-4839-9033-309e1f33584b*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_205474_205650*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
šn
˛
"__inference__traced_restore_207154
file_prefix;
(assignvariableop_embedding_16_embeddings:	' 4
"assignvariableop_1_dense_16_kernel: .
 assignvariableop_2_dense_16_bias:A
.assignvariableop_3_lstm_16_lstm_cell_16_kernel:	 K
8assignvariableop_4_lstm_16_lstm_cell_16_recurrent_kernel:	 ;
,assignvariableop_5_lstm_16_lstm_cell_16_bias:	&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: E
2assignvariableop_15_adam_embedding_16_embeddings_m:	' <
*assignvariableop_16_adam_dense_16_kernel_m: 6
(assignvariableop_17_adam_dense_16_bias_m:I
6assignvariableop_18_adam_lstm_16_lstm_cell_16_kernel_m:	 S
@assignvariableop_19_adam_lstm_16_lstm_cell_16_recurrent_kernel_m:	 C
4assignvariableop_20_adam_lstm_16_lstm_cell_16_bias_m:	E
2assignvariableop_21_adam_embedding_16_embeddings_v:	' <
*assignvariableop_22_adam_dense_16_kernel_v: 6
(assignvariableop_23_adam_dense_16_bias_v:I
6assignvariableop_24_adam_lstm_16_lstm_cell_16_kernel_v:	 S
@assignvariableop_25_adam_lstm_16_lstm_cell_16_recurrent_kernel_v:	 C
4assignvariableop_26_adam_lstm_16_lstm_cell_16_bias_v:	
identity_28˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ś
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ť
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_embedding_16_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_16_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_16_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_lstm_16_lstm_cell_16_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp8assignvariableop_4_lstm_16_lstm_cell_16_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp,assignvariableop_5_lstm_16_lstm_cell_16_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_embedding_16_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_16_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_16_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_lstm_16_lstm_cell_16_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ą
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_lstm_16_lstm_cell_16_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_16_lstm_cell_16_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ł
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_embedding_16_embeddings_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_16_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_16_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_16_lstm_cell_16_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ą
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_16_lstm_cell_16_recurrent_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_16_lstm_cell_16_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ą
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ç@
Ě
)__inference_gpu_lstm_with_fallback_206760

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d494a307-dced-4269-b473-1a192b8f604d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias

ˇ
(__inference_lstm_16_layer_call_fn_205201
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
identity˘StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_203074o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
"
_user_specified_name
inputs/0
ž;
Ŕ
 __inference_standard_lstm_201921

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_201835*
condR
while_cond_201834*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d7440df5-77b8-40d8-b129-4dc513ddf685*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ôK
Ą
'__forward_gpu_lstm_with_fallback_206507

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_2da92c77-19db-4d69-ae78-abd81b86f8ac*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_206332_206508*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_203361_203537
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_f43cc504-968b-46c4-93a8-e6ba5dafc816*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_203536*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_204500_204676
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_912539c9-9430-40df-bd9c-e69ba5214b52*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_204675*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
÷9

!__inference__wrapped_model_202201
embedding_16_inputE
2sequential_16_embedding_16_embedding_lookup_201762:	' E
2sequential_16_lstm_16_read_readvariableop_resource:	 G
4sequential_16_lstm_16_read_1_readvariableop_resource:	 C
4sequential_16_lstm_16_read_2_readvariableop_resource:	G
5sequential_16_dense_16_matmul_readvariableop_resource: D
6sequential_16_dense_16_biasadd_readvariableop_resource:
identity˘-sequential_16/dense_16/BiasAdd/ReadVariableOp˘,sequential_16/dense_16/MatMul/ReadVariableOp˘+sequential_16/embedding_16/embedding_lookup˘)sequential_16/lstm_16/Read/ReadVariableOp˘+sequential_16/lstm_16/Read_1/ReadVariableOp˘+sequential_16/lstm_16/Read_2/ReadVariableOp
sequential_16/embedding_16/CastCastembedding_16_input*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙°
+sequential_16/embedding_16/embedding_lookupResourceGather2sequential_16_embedding_16_embedding_lookup_201762#sequential_16/embedding_16/Cast:y:0*
Tindices0*E
_class;
97loc:@sequential_16/embedding_16/embedding_lookup/201762*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0ü
4sequential_16/embedding_16/embedding_lookup/IdentityIdentity4sequential_16/embedding_16/embedding_lookup:output:0*
T0*E
_class;
97loc:@sequential_16/embedding_16/embedding_lookup/201762*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ Ŕ
6sequential_16/embedding_16/embedding_lookup/Identity_1Identity=sequential_16/embedding_16/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ­
!sequential_16/dropout_16/IdentityIdentity?sequential_16/embedding_16/embedding_lookup/Identity_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
sequential_16/lstm_16/ShapeShape*sequential_16/dropout_16/Identity:output:0*
T0*
_output_shapes
:s
)sequential_16/lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_16/lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_16/lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#sequential_16/lstm_16/strided_sliceStridedSlice$sequential_16/lstm_16/Shape:output:02sequential_16/lstm_16/strided_slice/stack:output:04sequential_16/lstm_16/strided_slice/stack_1:output:04sequential_16/lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_16/lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ľ
"sequential_16/lstm_16/zeros/packedPack,sequential_16/lstm_16/strided_slice:output:0-sequential_16/lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_16/lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ž
sequential_16/lstm_16/zerosFill+sequential_16/lstm_16/zeros/packed:output:0*sequential_16/lstm_16/zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ h
&sequential_16/lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : š
$sequential_16/lstm_16/zeros_1/packedPack,sequential_16/lstm_16/strided_slice:output:0/sequential_16/lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_16/lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
sequential_16/lstm_16/zeros_1Fill-sequential_16/lstm_16/zeros_1/packed:output:0,sequential_16/lstm_16/zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
)sequential_16/lstm_16/Read/ReadVariableOpReadVariableOp2sequential_16_lstm_16_read_readvariableop_resource*
_output_shapes
:	 *
dtype0
sequential_16/lstm_16/IdentityIdentity1sequential_16/lstm_16/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Ą
+sequential_16/lstm_16/Read_1/ReadVariableOpReadVariableOp4sequential_16_lstm_16_read_1_readvariableop_resource*
_output_shapes
:	 *
dtype0
 sequential_16/lstm_16/Identity_1Identity3sequential_16/lstm_16/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 
+sequential_16/lstm_16/Read_2/ReadVariableOpReadVariableOp4sequential_16_lstm_16_read_2_readvariableop_resource*
_output_shapes	
:*
dtype0
 sequential_16/lstm_16/Identity_2Identity3sequential_16/lstm_16/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ú
%sequential_16/lstm_16/PartitionedCallPartitionedCall*sequential_16/dropout_16/Identity:output:0$sequential_16/lstm_16/zeros:output:0&sequential_16/lstm_16/zeros_1:output:0'sequential_16/lstm_16/Identity:output:0)sequential_16/lstm_16/Identity_1:output:0)sequential_16/lstm_16/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_201921˘
,sequential_16/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ż
sequential_16/dense_16/MatMulMatMul.sequential_16/lstm_16/PartitionedCall:output:04sequential_16/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
-sequential_16/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ť
sequential_16/dense_16/BiasAddBiasAdd'sequential_16/dense_16/MatMul:product:05sequential_16/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential_16/dense_16/SoftmaxSoftmax'sequential_16/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
IdentityIdentity(sequential_16/dense_16/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ű
NoOpNoOp.^sequential_16/dense_16/BiasAdd/ReadVariableOp-^sequential_16/dense_16/MatMul/ReadVariableOp,^sequential_16/embedding_16/embedding_lookup*^sequential_16/lstm_16/Read/ReadVariableOp,^sequential_16/lstm_16/Read_1/ReadVariableOp,^sequential_16/lstm_16/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2^
-sequential_16/dense_16/BiasAdd/ReadVariableOp-sequential_16/dense_16/BiasAdd/ReadVariableOp2\
,sequential_16/dense_16/MatMul/ReadVariableOp,sequential_16/dense_16/MatMul/ReadVariableOp2Z
+sequential_16/embedding_16/embedding_lookup+sequential_16/embedding_16/embedding_lookup2V
)sequential_16/lstm_16/Read/ReadVariableOp)sequential_16/lstm_16/Read/ReadVariableOp2Z
+sequential_16/lstm_16/Read_1/ReadVariableOp+sequential_16/lstm_16/Read_1/ReadVariableOp2Z
+sequential_16/lstm_16/Read_2/ReadVariableOp+sequential_16/lstm_16/Read_2/ReadVariableOp:d `
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
,
_user_specified_nameembedding_16_input
ž;
Ŕ
 __inference_standard_lstm_205808

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_205722*
condR
while_cond_205721*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_31626796-2eeb-455b-b989-31f5c67391d1*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_202895

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_86580efb-f15a-4581-9517-8d7c58130198*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
č

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_204061

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @q
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ľ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ł
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ |
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ v
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ f
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ôK
Ą
'__forward_gpu_lstm_with_fallback_202191

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d7440df5-77b8-40d8-b129-4dc513ddf685*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_202016_202192*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
â=
˛
__inference__traced_save_207063
file_prefix6
2savev2_embedding_16_embeddings_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop:
6savev2_lstm_16_lstm_cell_16_kernel_read_readvariableopD
@savev2_lstm_16_lstm_cell_16_recurrent_kernel_read_readvariableop8
4savev2_lstm_16_lstm_cell_16_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_embedding_16_embeddings_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableopA
=savev2_adam_lstm_16_lstm_cell_16_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_16_lstm_cell_16_bias_m_read_readvariableop=
9savev2_adam_embedding_16_embeddings_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableopA
=savev2_adam_lstm_16_lstm_cell_16_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_16_lstm_cell_16_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ś
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHĽ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ľ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_16_embeddings_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop6savev2_lstm_16_lstm_cell_16_kernel_read_readvariableop@savev2_lstm_16_lstm_cell_16_recurrent_kernel_read_readvariableop4savev2_lstm_16_lstm_cell_16_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_embedding_16_embeddings_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop=savev2_adam_lstm_16_lstm_cell_16_kernel_m_read_readvariableopGsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_16_lstm_cell_16_bias_m_read_readvariableop9savev2_adam_embedding_16_embeddings_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop=savev2_adam_lstm_16_lstm_cell_16_kernel_v_read_readvariableopGsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_16_lstm_cell_16_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ó
_input_shapesÁ
ž: :	' : ::	 :	 :: : : : : : : : : :	' : ::	 :	 ::	' : ::	 :	 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	' :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	' :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	' :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: 

d
F__inference_dropout_16_layer_call_and_return_conditional_losses_203109

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ôK
Ą
'__forward_gpu_lstm_with_fallback_204675

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_912539c9-9430-40df-bd9c-e69ba5214b52*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_204500_204676*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ř
Č
I__inference_sequential_16_layer_call_and_return_conditional_losses_204111

inputs&
embedding_16_204094:	' !
lstm_16_204098:	 !
lstm_16_204100:	 
lstm_16_204102:	!
dense_16_204105: 
dense_16_204107:
identity˘ dense_16/StatefulPartitionedCall˘"dropout_16/StatefulPartitionedCall˘$embedding_16/StatefulPartitionedCall˘lstm_16/StatefulPartitionedCallö
$embedding_16/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_16_204094*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_16_layer_call_and_return_conditional_losses_203100
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall-embedding_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_204061Ł
lstm_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0lstm_16_204098lstm_16_204100lstm_16_204102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_204032
 dense_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0dense_16_204105dense_16_204107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_203558x
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙×
NoOpNoOp!^dense_16/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

.__inference_sequential_16_layer_call_fn_204242

inputs
unknown:	' 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_204111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů)
Ď
while_body_204769
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
ôK
Ą
'__forward_gpu_lstm_with_fallback_202631

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d8412ebb-b775-4f9b-a70b-ce15dad8ece0*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_202456_202632*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_202015

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d7440df5-77b8-40d8-b129-4dc513ddf685*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_202016_202192
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_d7440df5-77b8-40d8-b129-4dc513ddf685*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_202191*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ôK
Ą
'__forward_gpu_lstm_with_fallback_206078

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0×
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ž
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_31626796-2eeb-455b-b989-31f5c67391d1*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_gpu_lstm_with_fallback_205903_206079*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ç@
Ě
)__inference_gpu_lstm_with_fallback_203853

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

:  Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:BÓ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ c

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_b63e776e-adf7-4312-a3b9-4155f169d032*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
Ů)
Ď
while_body_206151
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
Ů)
Ď
while_body_201835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_202456_202632
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_d8412ebb-b775-4f9b-a70b-ce15dad8ece0*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_202631*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
˛	

.__inference_sequential_16_layer_call_fn_203580
embedding_16_input
unknown:	' 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_203565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
,
_user_specified_nameembedding_16_input

ľ
(__inference_lstm_16_layer_call_fn_205223

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
identity˘StatefulPartitionedCallĺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_16_layer_call_and_return_conditional_losses_204032o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ů)
Ď
while_body_204319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:

d
F__inference_dropout_16_layer_call_and_return_conditional_losses_205167

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
ĺ	
Ś
H__inference_embedding_16_layer_call_and_return_conditional_losses_205152

inputs*
embedding_lookup_205146:	' 
identity˘embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ä
embedding_lookupResourceGatherembedding_lookup_205146Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/205146*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
dtype0Ť
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/205146*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

.__inference_sequential_16_layer_call_fn_204225

inputs
unknown:	' 
	unknown_0:	 
	unknown_1:	 
	unknown_2:	
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_203565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ž;
Ŕ
 __inference_standard_lstm_202361

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_202275*
condR
while_cond_202274*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_d8412ebb-b775-4f9b-a70b-ce15dad8ece0*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ž;
Ŕ
 __inference_standard_lstm_203759

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_203673*
condR
while_cond_203672*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_b63e776e-adf7-4312-a3b9-4155f169d032*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ž;
Ŕ
 __inference_standard_lstm_205379

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_205293*
condR
while_cond_205292*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_a972921d-1e9d-4839-9033-309e1f33584b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
ž;
Ŕ
 __inference_standard_lstm_206237

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ś
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ĺ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ś
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_206151*
condR
while_cond_206150*b
output_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :	 :	 :*=
api_implements+)lstm_2da92c77-19db-4d69-ae78-abd81b86f8ac*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_h:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinit_c:GC

_output_shapes
:	 
 
_user_specified_namekernel:QM

_output_shapes
:	 
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:

_user_specified_namebias
	
Á
while_cond_202714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_202714___redundant_placeholder04
0while_while_cond_202714___redundant_placeholder14
0while_while_cond_202714___redundant_placeholder24
0while_while_cond_202714___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ů)
Ď
while_body_206580
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    Ś
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Č
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ "*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : : :	 :	 :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :%	!

_output_shapes
:	 :!


_output_shapes	
:
§
˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_204032

inputs/
read_readvariableop_resource:	 1
read_1_readvariableop_resource:	 -
read_2_readvariableop_resource:	

identity_3˘Read/ReadVariableOp˘Read_1/ReadVariableOp˘Read_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	 *
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	 u
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 *
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:˛
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_lstm_203759i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_205903_206079
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_31626796-2eeb-455b-b989-31f5c67391d1*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_206078*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ď

-__inference_embedding_16_layer_call_fn_205142

inputs
unknown:	' 
identity˘StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_16_layer_call_and_return_conditional_losses_203100|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Â
ĺ
;__inference___backward_gpu_lstm_with_fallback_203854_204030
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ş
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:¤
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ u
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:¨
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ĺ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:˘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ť
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Ů
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Ĺ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:É
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ř
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ě
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:đ
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ď
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: ď
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: ň
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ą
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        §
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ł
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ś
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:ľ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:ˇ
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:ˇ
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:ˇ
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:ˇ
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:ˇ
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:ˇ
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:ˇ
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  ç
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Ż
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 ľ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ę
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Ň
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ö
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	 h

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapeső
ň:˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ::˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ :B::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ : ::::::::: : : : *=
api_implements+)lstm_b63e776e-adf7-4312-a3b9-4155f169d032*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_lstm_with_fallback_204029*
go_backwards( *

time_major( :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ ::6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :	

_output_shapes
:::
6
4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :1-
+
_output_shapes
:˙˙˙˙˙˙˙˙˙ :!

_output_shapes	
:B: 

_output_shapes
::-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "żL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ę
serving_defaultś
Z
embedding_16_inputD
$serving_default_embedding_16_input:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙<
dense_160
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Đ˛
č
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
ź
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Ú
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator
#cell
$
state_spec"
_tf_keras_rnn_layer
ť
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
J
0
-1
.2
/3
+4
,5"
trackable_list_wrapper
J
0
-1
.2
/3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
î
5trace_0
6trace_1
7trace_2
8trace_32
.__inference_sequential_16_layer_call_fn_203580
.__inference_sequential_16_layer_call_fn_204225
.__inference_sequential_16_layer_call_fn_204242
.__inference_sequential_16_layer_call_fn_204143Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z5trace_0z6trace_1z7trace_2z8trace_3
Ú
9trace_0
:trace_1
;trace_2
<trace_32ď
I__inference_sequential_16_layer_call_and_return_conditional_losses_204685
I__inference_sequential_16_layer_call_and_return_conditional_losses_205135
I__inference_sequential_16_layer_call_and_return_conditional_losses_204163
I__inference_sequential_16_layer_call_and_return_conditional_losses_204183Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z9trace_0z:trace_1z;trace_2z<trace_3
×BÔ
!__inference__wrapped_model_202201embedding_16_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ë
=iter

>beta_1

?beta_2
	@decay
Alearning_ratem+m,m-m.m/mv+v,v-v.v/v"
	optimizer
,
Bserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ń
Htrace_02Ô
-__inference_embedding_16_layer_call_fn_205142˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zHtrace_0

Itrace_02ď
H__inference_embedding_16_layer_call_and_return_conditional_losses_205152˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zItrace_0
*:(	' 2embedding_16/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Č
Otrace_0
Ptrace_12
+__inference_dropout_16_layer_call_fn_205157
+__inference_dropout_16_layer_call_fn_205162´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zOtrace_0zPtrace_1
ţ
Qtrace_0
Rtrace_12Ç
F__inference_dropout_16_layer_call_and_return_conditional_losses_205167
F__inference_dropout_16_layer_call_and_return_conditional_losses_205179´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zQtrace_0zRtrace_1
"
_generic_user_object
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
š

Sstates
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ë
Ytrace_0
Ztrace_1
[trace_2
\trace_32
(__inference_lstm_16_layer_call_fn_205190
(__inference_lstm_16_layer_call_fn_205201
(__inference_lstm_16_layer_call_fn_205212
(__inference_lstm_16_layer_call_fn_205223Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zYtrace_0zZtrace_1z[trace_2z\trace_3
×
]trace_0
^trace_1
_trace_2
`trace_32ě
C__inference_lstm_16_layer_call_and_return_conditional_losses_205652
C__inference_lstm_16_layer_call_and_return_conditional_losses_206081
C__inference_lstm_16_layer_call_and_return_conditional_losses_206510
C__inference_lstm_16_layer_call_and_return_conditional_losses_206939Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z]trace_0z^trace_1z_trace_2z`trace_3
"
_generic_user_object
ř
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator
h
state_size

-kernel
.recurrent_kernel
/bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
í
ntrace_02Đ
)__inference_dense_16_layer_call_fn_206948˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zntrace_0

otrace_02ë
D__inference_dense_16_layer_call_and_return_conditional_losses_206959˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zotrace_0
!: 2dense_16/kernel
:2dense_16/bias
.:,	 2lstm_16/lstm_cell_16/kernel
8:6	 2%lstm_16/lstm_cell_16/recurrent_kernel
(:&2lstm_16/lstm_cell_16/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_16_layer_call_fn_203580embedding_16_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Bý
.__inference_sequential_16_layer_call_fn_204225inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Bý
.__inference_sequential_16_layer_call_fn_204242inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
.__inference_sequential_16_layer_call_fn_204143embedding_16_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
I__inference_sequential_16_layer_call_and_return_conditional_losses_204685inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
I__inference_sequential_16_layer_call_and_return_conditional_losses_205135inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
§B¤
I__inference_sequential_16_layer_call_and_return_conditional_losses_204163embedding_16_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
§B¤
I__inference_sequential_16_layer_call_and_return_conditional_losses_204183embedding_16_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÖBÓ
$__inference_signature_wrapper_204208embedding_16_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBŢ
-__inference_embedding_16_layer_call_fn_205142inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
üBů
H__inference_embedding_16_layer_call_and_return_conditional_losses_205152inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ńBî
+__inference_dropout_16_layer_call_fn_205157inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ńBî
+__inference_dropout_16_layer_call_fn_205162inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
F__inference_dropout_16_layer_call_and_return_conditional_losses_205167inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
F__inference_dropout_16_layer_call_and_return_conditional_losses_205179inputs"´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
(__inference_lstm_16_layer_call_fn_205190inputs/0"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
(__inference_lstm_16_layer_call_fn_205201inputs/0"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
(__inference_lstm_16_layer_call_fn_205212inputs"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
(__inference_lstm_16_layer_call_fn_205223inputs"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŹBŠ
C__inference_lstm_16_layer_call_and_return_conditional_losses_205652inputs/0"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŹBŠ
C__inference_lstm_16_layer_call_and_return_conditional_losses_206081inputs/0"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŞB§
C__inference_lstm_16_layer_call_and_return_conditional_losses_206510inputs"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŞB§
C__inference_lstm_16_layer_call_and_return_conditional_losses_206939inputs"Ő
Ě˛Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
5
-0
.1
/2"
trackable_list_wrapper
5
-0
.1
/2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ä2Áž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ä2Áž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_16_layer_call_fn_206948inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
řBő
D__inference_dense_16_layer_call_and_return_conditional_losses_206959inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
N
w	variables
x	keras_api
	ytotal
	zcount"
_tf_keras_metric
^
{	variables
|	keras_api
	}total
	~count

_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
.
}0
~1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-	' 2Adam/embedding_16/embeddings/m
&:$ 2Adam/dense_16/kernel/m
 :2Adam/dense_16/bias/m
3:1	 2"Adam/lstm_16/lstm_cell_16/kernel/m
=:;	 2,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m
-:+2 Adam/lstm_16/lstm_cell_16/bias/m
/:-	' 2Adam/embedding_16/embeddings/v
&:$ 2Adam/dense_16/kernel/v
 :2Adam/dense_16/bias/v
3:1	 2"Adam/lstm_16/lstm_cell_16/kernel/v
=:;	 2,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v
-:+2 Adam/lstm_16/lstm_cell_16/bias/vŠ
!__inference__wrapped_model_202201-./+,D˘A
:˘7
52
embedding_16_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
dense_16"
dense_16˙˙˙˙˙˙˙˙˙¤
D__inference_dense_16_layer_call_and_return_conditional_losses_206959\+,/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙ 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 |
)__inference_dense_16_layer_call_fn_206948O+,/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙ 
Ş "˙˙˙˙˙˙˙˙˙Ŕ
F__inference_dropout_16_layer_call_and_return_conditional_losses_205167v@˘=
6˘3
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
p 
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 Ŕ
F__inference_dropout_16_layer_call_and_return_conditional_losses_205179v@˘=
6˘3
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
p
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
+__inference_dropout_16_layer_call_fn_205157i@˘=
6˘3
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
p 
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
+__inference_dropout_16_layer_call_fn_205162i@˘=
6˘3
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
p
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ ˝
H__inference_embedding_16_layer_call_and_return_conditional_losses_205152q8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "2˘/
(%
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
-__inference_embedding_16_layer_call_fn_205142d8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "%"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ Ä
C__inference_lstm_16_layer_call_and_return_conditional_losses_205652}-./O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙ 
 Ä
C__inference_lstm_16_layer_call_and_return_conditional_losses_206081}-./O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙ 
 ˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_206510v-./H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙ 
 ˝
C__inference_lstm_16_layer_call_and_return_conditional_losses_206939v-./H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙ 
 
(__inference_lstm_16_layer_call_fn_205190p-./O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p 

 
Ş "˙˙˙˙˙˙˙˙˙ 
(__inference_lstm_16_layer_call_fn_205201p-./O˘L
E˘B
41
/,
inputs/0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p

 
Ş "˙˙˙˙˙˙˙˙˙ 
(__inference_lstm_16_layer_call_fn_205212i-./H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p 

 
Ş "˙˙˙˙˙˙˙˙˙ 
(__inference_lstm_16_layer_call_fn_205223i-./H˘E
>˘;
-*
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 

 
p

 
Ş "˙˙˙˙˙˙˙˙˙ Ę
I__inference_sequential_16_layer_call_and_return_conditional_losses_204163}-./+,L˘I
B˘?
52
embedding_16_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 Ę
I__inference_sequential_16_layer_call_and_return_conditional_losses_204183}-./+,L˘I
B˘?
52
embedding_16_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ž
I__inference_sequential_16_layer_call_and_return_conditional_losses_204685q-./+,@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ž
I__inference_sequential_16_layer_call_and_return_conditional_losses_205135q-./+,@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ˘
.__inference_sequential_16_layer_call_fn_203580p-./+,L˘I
B˘?
52
embedding_16_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙˘
.__inference_sequential_16_layer_call_fn_204143p-./+,L˘I
B˘?
52
embedding_16_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_16_layer_call_fn_204225d-./+,@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
.__inference_sequential_16_layer_call_fn_204242d-./+,@˘=
6˘3
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙Â
$__inference_signature_wrapper_204208-./+,Z˘W
˘ 
PŞM
K
embedding_16_input52
embedding_16_input˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"3Ş0
.
dense_16"
dense_16˙˙˙˙˙˙˙˙˙