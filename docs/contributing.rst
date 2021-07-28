.. _contributing:

============
Contributing
============

ElikoPy is an open source project, meaning we have the challenge of limited 
resources. We are grateful for any support that you can offer. Helping other 
users, raising issues, helping write documentation, or contributing code are all 
ways to help!

--------------
Raise an Issue
--------------

For general bugs/issues, you can open an issue `at the GitHub repo 
<https://github.com/Hyedryn/elikopy/issues/new>`_.

-------------------
Write Documentation
-------------------

We (like almost all open source software providers) have a documentation 
dilemma… We tend to focus on the code features and functionality before working 
on documentation.

You can contribute to the documentation by `raising an issue to suggest an 
improvement <https://github.com/Hyedryn/elikopy/issues/new>`_ or by 
sending a `pull request 
<https://github.com/Hyedryn/elikopy/compare>`_ on `our repository 
<https://github.com/Hyedryn/elikopy>`_.

.. _contribute-to-the-code:

----------------------
Contribute to the code
----------------------

We use the traditional 
`GitHub Flow <https://guides.github.com/introduction/flow/>`_ to develop. This 
means that you fork the main repo, create a new branch to make changes, and
submit a pull request (PR) to the master branch.

Step 1. Fork the repo
=====================

To contribute to ElikoPy, you should obtain a GitHub account and fork the 
`ElikoPy <https://github.com/Hyedryn/elikopy>`_ repository. Once forked, 
clone your fork of the repo to your computer. (Obviously, you should replace 
``your-username`` with your GitHub username.)

.. code-block:: none

    $ git clone https://github.com/your-username/elikopy.git && \
        cd elikopy/

Step 2. Checkout a new branch
=============================

`Branches <https://guides.github.com/introduction/flow//>`_ are a way of 
isolating your features from the main branch. Given that we’ve just cloned the 
repo, we will probably want to make a new branch from master in which to work on
our new feature. Lets call that branch ``new-feature``:

.. code-block:: none

    $ git checkout master && \
        git checkout -b new-feature

.. note::

    You can always check which branch you are in by running ``git branch``.

Step 3. Make your changes
=========================

On your new branch, go nuts! Make changes, test them, and when you are happy 
commit the changes to the branch:

.. code-block:: none

    $ git add file-changed1 file-changed2...

    $ git commit -m "what changed?"

This commit message is important - it should describe exactly the changes that 
you have made. Good commit messages read like so:

.. code-block:: none

    $ git commit -m "changed function preproc in core.py to output new mask to fix #2"

    $ git commit -m "updated docs about MF to close #10"

The tags ``close #10`` and ``fix #2`` are referencing issues that are posted on 
the upstream repo where you will direct your pull request. When your PR is 
merged into the master branch, these messages will automatically close the 
issues, and further, they will link your commits directly to the issues they 
intend to fix. This will help future maintainers understand your contribution, 
or (hopefully not) revert the code back to a previous version if necessary.

Step 4. Push your branch to your fork
=====================================

When you are done with your commits, you should push your branch to your fork 
(and you can also continuously push commits here as you work):

.. code-block:: none

    $ git push origin new-feature


Note that you should always check the status of your branches to see what has 
been pushed (or not):

.. code-block:: none

    $ git status


Step 5. Submit a Pull Request
=============================

Once you have pushed your branch, then you can go to your fork (in the web GUI 
on GitHub) and `submit a Pull Request
<https://help.github.com/articles/creating-a-pull-request/>`_. Regardless of the 
name of your branch, your PR should be submitted to the ElikoPy ``master`` 
branch. Submitting your PR will open a conversation thread for the maintainers 
of ElikoPy to discuss your contribution. At this time, the continuous 
integration that is linked with the code base will also be executed. If there is 
an issue, or if the maintainers suggest changes, you can continue to push 
commits to your branch and they will update the Pull Request.

Step 6. Keep your branch in sync
================================

Cloning the repo will create an exact copy of the ElikoPy repository at that 
moment. As you work, your branch may become out of date as others merge changes
into the upstream master. In the event that you need to update a branch, you 
will need to follow the next steps:

.. code-block:: none

    $ git remote add upstream https://github.com/Hyedryn/elikopy.git && # to add a new remote named "upstream" \
      git checkout master && # or another branch to be updated \
      git pull upstream master && \
      git push origin master && # to update your fork \
      git checkout new-feature && \
      git merge master 







