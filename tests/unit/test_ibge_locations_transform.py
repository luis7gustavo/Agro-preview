from agri_decision.transforms.ibge_locations import build_location_records


def test_location_dimension_uses_official_geometry_for_centroid() -> None:
    # Synthetic geometry fixture used only to validate transformation behavior.
    municipalities = [
        {
            "id": 1234567,
            "nome": "Município de Teste",
            "regiao-imediata": {
                "regiao-intermediaria": {
                    "UF": {
                        "sigla": "DF",
                        "regiao": {"nome": "Centro-Oeste"},
                    }
                }
            },
        }
    ]
    meshes = {
        "DF": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"codarea": "1234567"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                    },
                }
            ],
        }
    }

    location = build_location_records(municipalities, meshes)[0]

    assert location.latitude_centroid == 1.0
    assert location.longitude_centroid == 1.0
    assert location.codigo_ibge == "1234567"


def test_location_without_published_mesh_is_preserved_as_missing() -> None:
    municipalities = [
        {
            "id": 1234567,
            "nome": "Município de Teste",
            "regiao-imediata": {
                "regiao-intermediaria": {
                    "UF": {
                        "sigla": "DF",
                        "regiao": {"nome": "Centro-Oeste"},
                    }
                }
            },
        }
    ]

    location = build_location_records(municipalities, {})[0]

    assert location.geometry is None
    assert location.latitude_centroid is None
    assert location.quality_flags == ["MISSING_OFFICIAL_GEOMETRY"]
