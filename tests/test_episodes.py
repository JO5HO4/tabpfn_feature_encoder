import numpy as np

from tabpfn_feature_encoder.training.episodes import RatioEpisodeSampler


def test_ratio_episode_sampler_splits_batch_in_half() -> None:
    y = np.asarray([0, 1] * 10)

    episode = RatioEpisodeSampler(support_query_ratio=0.5, random_state=1).sample(y)

    assert len(episode.support_idx) == 10
    assert len(episode.query_idx) == 10
    assert set(episode.support_idx).isdisjoint(set(episode.query_idx))
    assert set(np.unique(y[episode.support_idx])) == {0, 1}
