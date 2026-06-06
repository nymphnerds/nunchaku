Z-Image
=======

The following is the example of running Nunchaku version of Z-Image text-to-image pipeline.

.. tabs::

   .. tab:: Z-Image-Turbo

      .. literalinclude:: ../../../examples/v1/z-image-turbo.py
         :language: python
         :caption: Running Z-Image-Turbo (`examples/v1/z-image-turbo.py <https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/z-image-turbo.py>`__)
         :linenos:

   .. tab:: Z-Image-Turbo ControlNet

      .. literalinclude:: ../../../examples/v1/z-image-controlnet.py
         :language: python
         :caption: Running Z-Image-Turbo ControlNet (`examples/v1/z-image-controlnet.py <https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/z-image-controlnet.py>`__)
         :linenos:


For more details, see :class:`~nunchaku.models.transformers.transformer_zimage.NunchakuZImageTransformer2DModel`.

.. note::
   Z-Image ControlNet requires a Diffusers build that provides ``ZImageControlNetPipeline`` and
   ``ZImageControlNetModel``. The ControlNet module itself runs at its original precision; Nunchaku
   replaces the Z-Image transformer.
