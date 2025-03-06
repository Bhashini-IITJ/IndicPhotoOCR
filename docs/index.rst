.. IndicPhotoOCR documentation master file, created by
   sphinx-quickstart on Thu Mar  6 19:20:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

IndicPhotoOCR Documentation
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


End-2-End OCR Class
^^^^^^^^^^^^^^^^^^^
.. automodule:: IndicPhotoOCR.ocr
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: crop_and_identify_script


Detection
^^^^^^^^^
.. automodule:: IndicPhotoOCR.detection.textbpn.textbpnpp_detector
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ensure_model, pad_image, rescale_result, to_device


Recognition
^^^^^^^^^^^

.. automodule:: IndicPhotoOCR.recognition.parseq_recogniser
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ensure_model, get_transform, get_model_output, load_model

Script Identification
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: IndicPhotoOCR.script_identification.vit.vit_infer
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: unzip_file, ensure_model

