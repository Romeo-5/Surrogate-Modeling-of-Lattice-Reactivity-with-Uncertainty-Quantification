# OpenMC + the analysis stack, pinned through conda-forge.
#
# Built on micromamba rather than the official openmc/openmc image so that the
# transport code and the surrogate-modelling stack are resolved together by one
# solver -- the two halves of this project have to share a Python environment,
# and pip-installing scikit-learn on top of a prebuilt OpenMC image is exactly
# where ABI mismatches show up.
#
# Cross-section data is deliberately NOT baked in: the ENDF/B HDF5 library is
# several GB and belongs in a mounted volume, not in an image layer.
FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV OPENMC_CROSS_SECTIONS=/data/endfb-vii.1-hdf5/cross_sections.xml \
    PYTHONPATH=/work/src \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

WORKDIR /work

# Fail at build time, not three hours into a campaign, if the stack is broken.
RUN python -c "import openmc, sklearn, scipy, pandas, matplotlib; print('openmc', openmc.__version__)"

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["bash"]
