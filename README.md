# Quantile Forecasts Using Neural Networks for Retail Inventory Management with Fixed-Time Replenishment Intervals

Master's thesis in Informatics, Graduate Program in Informatics (PPGI), Federal University of Amazonas (UFAM), Manaus, Brazil. This repository holds the text and slides of the **preliminary defense** (qualifying exam), August 2019.

- **Author:** Willians Cassiano de Freitas Abreu
- **Advisor:** Prof. Marco Antônio Pinheiro de Cristo
- **Co-advisor:** Prof. Juan Gabriel Colonna

## Files

| File | Description |
|---|---|
| [Master's Thesis Text Preliminary Defense.pdf](Master's%20Thesis%20Text%20Preliminary%20Defense.pdf) | Thesis text (English, 58 pages) |
| [Master's Thesis Preliminary Defense.pdf](Master's%20Thesis%20Preliminary%20Defense.pdf) | Defense slides (English, 27 slides) |

## Summary

A retailer has to keep enough stock to meet customer demand while suppliers take time to deliver (the lead time). Because sales are uncertain, retailers keep safety stock, which creates overstocking costs: holding costs, opportunity costs, and losses from damaged, expired and stolen items. Keeping too little stock causes stock-outs, which can cost even more than overstocking depending on the product segment.

The work addresses both problems together. It builds a mathematical model of the financial loss from stock-outs and overstocking in a setting where **demand and lead times are stochastic and replenishment happens at fixed intervals**. It then trains neural networks to minimize that loss directly, instead of forecasting demand and deriving an order quantity from the forecast.

Main ideas:

- **Service levels as quantile forecasts.** The target service level of an item maps to a quantile of the demand distribution over the lead time. The models are trained with the **pinball (quantile) loss**.
- **Optimizing the business cost directly.** The thesis proposes an objective function for gradient descent that encodes the cost of overstocking and stock-outs, so the network learns the order policy itself.
- **Case study on real retail data.** The case study uses daily sales from a Brazilian retailer, focused on the best-selling items (the "A1" segment). For example, the top 500 items account for about half of revenue.
- **A simulator of retailer mechanics.** It replays real demand and lead times to compare the neural network policy against a classical baseline and against the optimal inventory series.

## Contents

1. **Introduction:** problem definition, research hypothesis and objectives.
2. **Theoretical foundations:** inventory management policies, uncertainty in demand and lead times, service levels and quantile forecasts, the pinball loss, optimizing the cost directly with neural networks, and related work.
3. **Neural networks for quantile forecasts:** the case study dataset, preprocessing, and the network architecture and training.
4. **Simulator, model evaluation and preliminary results.**
5. **Open problems and next steps:** aggregated pinball loss, temporal connectionist models, extending to the B and C segments, and the schedule.

**Keywords:** inventory management, deep learning, quantile forecasting.
