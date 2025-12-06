import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def build_unet(input_shape=(256, 256, 1),
                        initial_filters=32,       # f_init
                        depth=4,                  # d
                        kernel_size=3,            # k
                        activation_name='relu',   # sigma
                        norm_type='batch',        # Norm
                        dropout_rate=0.0,         # p_drop
                        use_bias=False,           # b_use
                        pooling_type='max',       # pool
                        upsample_type='transpose' # Estrategia Up-Sampling
                        ):
    """
    Construye una U-Net parametrizable basada en la Tabla 7.1 de optimización evolutiva.
    """

    # --- 1. Helper para Normalización ---
    def get_norm_layer(norm_type_str):
        # SI ES NONE, DEVOLVER IDENTIDAD INMEDIATAMENTE
        if norm_type_str is None:
            return layers.Lambda(lambda x: x)
            
        # Convertir a string de forma segura
        nt = str(norm_type_str).lower()
        
        if nt == 'batch':
            return layers.BatchNormalization()
        elif nt == 'instance':
            return layers.GroupNormalization(groups=-1)
        elif nt == 'group':
            # GroupNorm por defecto (seguro para pocos canales)
            return layers.GroupNormalization(groups=4)
        elif nt == 'none': # Por si llega como string "None"
            return layers.Lambda(lambda x: x)
        else:
            return layers.Lambda(lambda x: x)

    # --- 2. Helper para Activación ---
    def get_activation_layer(act_name):
        an = act_name.lower()
        if an == 'relu':
            return layers.ReLU()
        elif an == 'elu':
            return layers.ELU()
        elif an == 'leakyrelu':
            return layers.LeakyReLU()
        elif an == 'gelu':
            # GELU suele estar disponible como función, la envolvemos en capa
            return layers.Activation(tf.keras.activations.gelu)
        elif an == 'swish':
            return layers.Activation(tf.keras.activations.swish)
        else:
            return layers.ReLU() # Fallback

    # --- 3. Helper para Pooling ---
    def get_pooling_layer(pool_name):
        pn = pool_name.lower()
        if pn == 'max':
            return layers.MaxPooling2D((2, 2))
        elif pn == 'average':
            return layers.AveragePooling2D((2, 2))
        else:
            return layers.MaxPooling2D((2, 2))

    # --- 4. Helper para el Bloque Convolucional (Conv -> Norm -> Act) ---
    def conv_block(input_tensor, n_filters):
        # Primera Convolución
        x = layers.Conv2D(n_filters, kernel_size, padding='same', use_bias=use_bias)(input_tensor)
        x = get_norm_layer(norm_type)(x)
        x = get_activation_layer(activation_name)(x)
        
        # Segunda Convolución
        x = layers.Conv2D(n_filters, kernel_size, padding='same', use_bias=use_bias)(x)
        x = get_norm_layer(norm_type)(x)
        x = get_activation_layer(activation_name)(x)
        return x

    # --- COMIENZO DE LA ARQUITECTURA ---
    
    inputs = layers.Input(shape=input_shape)
    skips = []
    x = inputs
    current_filters = initial_filters

    # --- ENCODER ---
    for i in range(depth):
        # Aplicar bloque convolucional
        c = conv_block(x, current_filters)
        
        # Guardar skip connection
        skips.append(c)
        
        # Down-sampling (Pooling)
        x = get_pooling_layer(pooling_type)(c)
        
        # Aumentar filtros
        current_filters *= 2

    # --- BOTTLENECK ---
    b = conv_block(x, current_filters)

    # Dropout (p_drop)
    if dropout_rate > 0.0:
        b = layers.Dropout(dropout_rate)(b)

    # --- DECODER ---
    x = b
    for i in reversed(range(depth)):
        current_filters //= 2
        
        # Estrategia de Up-Sampling
        if upsample_type.lower() == 'transpose':
            # Opción A: Conv2DTranspose (Aprendible)
            u = layers.Conv2DTranspose(current_filters, (2, 2), strides=(2, 2), padding='same')(x)
        else:
            # Opción B: Bilinear Upsample + Conv (Fija + Ajuste de canales)
            # Primero escalamos la imagen
            u = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
            # Luego usamos una conv 1x1 para reducir el número de canales a 'current_filters'
            # para poder concatenar correctamente con la skip connection
            u = layers.Conv2D(current_filters, (1, 1), padding='same', use_bias=use_bias)(u)

        # Concatenación
        u = layers.concatenate([u, skips[i]])
        
        # Bloque Convolucional del Decoder
        x = conv_block(u, current_filters)

    # --- SALIDA ---
    # Salida binaria (sigmoide) para segmentación
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model