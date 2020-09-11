All scripts
===========

.. automodule:: mcot.core._scripts

    .. rubric:: Link directory contents
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        iter_link
        merge_hcp

    .. rubric:: NIFTI/GIFTI/CIFTI to dataframes
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        to_dataframe
        to_dataframe_tree

    .. rubric:: create/analyze CIFTI files
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        cifti.convert
        cifti.correlate
        cifti.ptx2dconn
        cifti.smooth

    .. rubric:: Functional MRI scripts
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        function.normalize

    .. rubric:: Gyral coordinate system
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        gcoord.gen
        gcoord.gui
        gcoord.mult
        gcoord.split
        gcoord.transition

    .. rubric:: ProbtrackX helpers
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        ptx.select_vertex

    .. rubric:: Splits voxel-wise job into multiple sub-jobs
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        split.merge
        split.run
        split.submit

    .. rubric:: GIFTI surfaces
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        surface.from_mask
        surface.from_ridge
        surface.gradient
        surface.roi_dist_gifti
        surface.smooth
        surface.watershed

    .. rubric:: Surface parcellations
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        parcel.cluster
        parcel.combine
        parcel.discretise
        parcel.random
        parcel.spectral

    .. rubric:: Tree helper functions
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        tree.extract

    .. rubric:: Various
    .. autosummary::
        :toctree: _scripts
        :template: script.rst

        round_bvals

