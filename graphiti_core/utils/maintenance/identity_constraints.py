"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from graphiti_core.utils.identity_constraints import (
    IDENTITY_FIELDS as IDENTITY_FIELDS,
    IDENTITY_TYPES as IDENTITY_TYPES,
    PROTECTED_ATTRIBUTES as PROTECTED_ATTRIBUTES,
    SCOPE_FIELDS as SCOPE_FIELDS,
    STRUCTURAL_ATTRIBUTES as STRUCTURAL_ATTRIBUTES,
    IdentityConflict as IdentityConflict,
    identities_compatible as identities_compatible,
    inferred_attributes as inferred_attributes,
    merge_provenance as merge_provenance,
    preserve_authority as preserve_authority,
    prompt_attributes as prompt_attributes,
    protected_attributes as protected_attributes,
    require_same_identity as require_same_identity,
    scopes_compatible as scopes_compatible,
)
