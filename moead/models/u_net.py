import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def build_unet(input_shape=(256, 256, 1),
               initial_filters=32,
               depth=4,
               kernel_size=3,
               activation_name='relu',
               norm_type='batch',  # Por defecto Batch
               dropout_rate=0.0,
               use_bias=False,
               pooling_type='max',
               upsample_type='transpose',
               **kwargs):

    # --- 1. HELPER DE NORMALIZACIÓN SIMPLIFICADO ---
    def get_norm_layer(norm_type_str):
        # Si es None, no ponemos nada
        if norm_type_str is None:
            return layers.Lambda(lambda x: x)
        
        # Para cualquier otro caso ('Batch', 'Group', 'Instance'),
        # devolvemos SIEMPRE BatchNormalization.
        # Así evitamos errores si el algoritmo genético intenta pedir otra cosa.
        return layers.BatchNormalization()

    # --- 2. HELPER DE ACTIVACIÓN ---
    def get_activation_layer(act_name):
        if act_name is None: return layers.ReLU()
        an = str(act_name).lower()
        
        if an == 'relu': return layers.ReLU()
        elif an == 'elu': return layers.ELU()
        elif an == 'leakyrelu': return layers.LeakyReLU()
        elif an == 'gelu': 
            try:
                return layers.Activation(tf.keras.activations.gelu)
            except AttributeError:
                return layers.ELU() 
        elif an == 'swish': 
            return layers.Activation(tf.keras.activations.swish)
        return layers.ReLU()

    # --- 3. HELPER DE POOLING ---
    def get_pooling_layer(pool_name):
        pn = str(pool_name).lower()
        if 'average' in pn: return layers.AveragePooling2D((2, 2))
        return layers.MaxPooling2D((2, 2))

    # --- 4. BLOQUE CONVOLUCIONAL ---
    def conv_block(input_tensor, n_filters):
        # Asegurar entero
        k_size = kernel_size
        if hasattr(k_size, 'item'): k_size = int(k_size)
        if isinstance(k_size, (list, tuple)): k_size = tuple(int(k) for k in k_size)
        else: k_size = (int(k_size), int(k_size))

        x = layers.Conv2D(n_filters, k_size, padding='same', use_bias=use_bias)(input_tensor)
        x = get_norm_layer(norm_type)(x)
        x = get_activation_layer(activation_name)(x)
        
        x = layers.Conv2D(n_filters, k_size, padding='same', use_bias=use_bias)(x)
        x = get_norm_layer(norm_type)(x)
        x = get_activation_layer(activation_name)(x)
        return x

    # --- CONSTRUCCIÓN ---
    inputs = layers.Input(shape=input_shape)
    skips = []
    x = inputs
    current_filters = int(initial_filters)

    # Encoder
    for i in range(int(depth)):
        c = conv_block(x, current_filters)
        skips.append(c)
        x = get_pooling_layer(pooling_type)(c)
        current_filters *= 2

    # Bottleneck
    b = conv_block(x, current_filters)
    if dropout_rate > 0.0:
        b = layers.Dropout(dropout_rate)(b)

    # Decoder
    x = b
    for i in reversed(range(int(depth))):
        current_filters //= 2
        
        if str(upsample_type).lower() == 'transposeconv':
            u = layers.Conv2DTranspose(current_filters, (2, 2), strides=(2, 2), padding='same')(x)
        else:
            u = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
            u = layers.Conv2D(current_filters, (1, 1), padding='same', use_bias=use_bias)(u)

        u = layers.concatenate([u, skips[i]])
        x = conv_block(u, current_filters)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(x)
    
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])